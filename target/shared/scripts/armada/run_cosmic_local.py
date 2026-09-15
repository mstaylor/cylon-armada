# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One rank of a local multi-rank Cosmic AI run (SP1 Task 6).

Launched once per rank by run_cosmic_local.sh. Builds the six-operator agentic
workflow, derives the connection topology from its compiled plan, opens an
FMIBridge over the direct-redis channel, and executes the pipeline.

Services are mocked by default: --live switches to real Bedrock/Redis/DynamoDB
and therefore costs money on every rank (one embedding call and one LLM call per
galaxy), so it is opt-in rather than the default.

Environment (set by the launcher): RANK, WORLD_SIZE, COMM_NAME, REDIS_HOST,
REDIS_PORT, FMI_LISTEN_PORT.

Rank identity differs by arm. Under the armada backend the rank is the one the
FMI channel negotiates through Redis INCR (bridge.rank), which can differ from
the RANK the launcher requested; under the langchain backend there is no
channel to negotiate through, so the rank is RANK itself. Both are used to pick
this rank's shard and to stamp the rank that originated each context, so the
two mechanisms are interchangeable for correctness — but do not assume a rank's
shard is the same under both arms of a local run, and never reuse COMM_NAME
across runs or the INCR counters collide.

The origination stamp is why the workflow is built after the bridge rather than
before it: a context tagged with the requested rank instead of the negotiated
one would be counted as ingested by the rank that actually created it, and
records_written would stop meaning the same thing on the two arms.
"""

import argparse
import json
import logging
import os
import sys
import time
from enum import Enum

import numpy as np
import pyarrow as pa

from armada.cosmic_workflow import build_cosmic_workflow
from armada.epochs import plan_epochs
from armada.executor import ArmadaExecutor, InputPlacement, required_peer_map
from armada.run_metrics import RunMetrics
from communicator.fmi_bridge import FMIBridge
from cosmic_ai.task_generator import generate_tasks_from_results

logger = logging.getLogger("run_cosmic_local")


def astromae_fixture(n_galaxies, seed=42):
    """Deterministic stand-in for an AstroMAE inference pass.

    Used when --real-inference is not passed, so the pipeline can be exercised
    without the weights or the SDSS partition present. Note its photometry is
    drawn on an apparent-magnitude scale, unlike the standardized features the
    real model consumes — see cosmic_ai.task_generator._format_bands.
    """
    rng = np.random.default_rng(seed)
    predictions = np.round(rng.uniform(0.15, 0.95, n_galaxies), 3)
    residuals = np.round(rng.normal(0.0, 0.06, n_galaxies), 3)
    true_redshifts = np.clip(predictions - residuals, 0.01, None)
    base = rng.uniform(17.0, 20.5, n_galaxies)
    magnitudes = np.round(base[:, None] - np.array([0.0, 0.7, 1.2, 1.5, 1.7]), 2)
    return predictions, true_redshifts, magnitudes


def shard_bounds(n_items, world_size, rank):
    """Contiguous [start, stop) slice of n_items for this rank.

    The first n_items % world_size ranks take one extra item, so an
    indivisible population still distributes completely — world sizes here are
    arbitrary, not just powers of two.
    """
    base, remainder = divmod(n_items, world_size)
    start = rank * base + min(rank, remainder)
    return start, start + base + (1 if rank < remainder else 0)


def astromae_inference(data_path, model_path, world_size, rank, batch_size, device,
                       n_galaxies=None):
    """Run AstroMAE over this rank's shard of the SDSS partition.

    Each rank infers only its own contiguous slice, which is what makes the
    workflow's Scatter redundant (see InputPlacement.PreDistributed) and
    mirrors the data-parallel inference of the original Cosmic AI runs.
    Returns the same triple as astromae_fixture, plus the inference metrics.
    """
    import torch

    from cosmic_ai.inference import load_data, load_model, run_inference

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"INFERENCE_DEVICE=cuda but torch reports no CUDA device (rank={rank}) — "
            f"refusing to fall back to CPU silently, which would make the recorded "
            f"device wrong for every measurement in this run"
        )

    dataset = load_data(data_path, device=device)
    total = len(dataset) if n_galaxies is None else min(n_galaxies, len(dataset))
    start, stop = shard_bounds(total, world_size, rank)
    logger.info("rank %d inferring galaxies [%d, %d) of %d on %s",
                rank, start, stop, total, device)

    model = load_model(model_path, device=device)
    shard = torch.utils.data.Subset(dataset, list(range(start, stop)))
    result = run_inference(model, shard, batch_size=batch_size, device=device)
    return (result["predictions"], result["true_redshifts"], result["magnitudes"],
            result["metrics"], (start, stop))


def mock_services(dimensions):
    from unittest.mock import MagicMock

    embedding_service = MagicMock()
    embedding_service.embed.side_effect = lambda text: (
        np.full(dimensions, float(len(text) % 7 + 1), dtype=np.float32),
        {"token_count": len(text)},
    )
    context_router = MagicMock()
    context_router.find_similar.return_value = []
    context_manager = MagicMock()
    context_manager.store_context.return_value = "mock-ctx"
    chain_executor = MagicMock()
    chain_executor.execute.side_effect = lambda prompt: {
        "response": f"[mock analysis] {prompt[:60]}",
        "input_tokens": 0, "output_tokens": 0, "latency_ms": 0.0,
        "model_id": "mock",
    }
    return embedding_service, context_router, context_manager, chain_executor


def live_services(dimensions):
    from chain.executor import ChainExecutor
    from context.embedding import EmbeddingService
    from context.manager import ContextManager
    from context.router import ContextRouter
    from cost.bedrock_pricing import BedrockConfig

    config = BedrockConfig.resolve()
    embedding_service = EmbeddingService(config=config)
    context_manager = ContextManager.from_config(
        config,
        redis_host=os.environ.get("REDIS_HOST", ""),
        redis_port=int(os.environ.get("REDIS_PORT", 6379)),
    )
    context_router = ContextRouter(context_manager, config=config)
    chain_executor = ChainExecutor(config=config)
    return embedding_service, context_router, context_manager, chain_executor


class ExecutionBackend(Enum):
    """Which runtime executes the workflow.

    Armada drives the compiled plan over the Cylon collectives. LangChain runs
    the same chain through Runnable.invoke() and shares context through the
    store instead of through a collective, which is what a LangChain
    deployment does and needs no bridge at all.
    """

    Armada = "armada"
    LangChain = "langchain"


_CONTEXT_STORE = {
    ExecutionBackend.Armada: "cylon",
    ExecutionBackend.LangChain: "redis",
}


def context_store_for(backend):
    """The context store an arm runs on. The arm is its store: Armada on the
    Arrow ContextTable, LangChain on Redis, so this is not independently
    tunable. An environment that already names a different store is refused
    rather than silently overridden.
    """
    required = _CONTEXT_STORE[backend]
    configured = os.environ.get("CONTEXT_BACKEND")
    if configured and configured != required:
        raise ValueError(
            f"CONTEXT_BACKEND={configured!r} conflicts with backend {backend.value!r}, "
            f"which runs on {required!r}"
        )
    return required


def run_epochs(seq, shards, backend, bridge, ctx, root=0):
    """Execute one epoch per shard, returning each epoch's result table.

    Under Armada each epoch is one executor pass with the input already in
    place on this rank, ending in MemoryUpsert's AllGather. Under LangChain it
    is a plain invoke(); nothing crosses ranks because the store is shared.
    """
    results = []
    for shard in shards:
        if backend is ExecutionBackend.LangChain:
            results.append(seq.invoke(shard))
        elif backend is ExecutionBackend.Armada:
            results.append(ArmadaExecutor(bridge).run(
                seq, input_tables=shard, ctx=ctx, root=root,
                placement=InputPlacement.PreDistributed))
        else:
            raise ValueError(f"unknown execution backend {backend!r}")
    return results


def write_record(path, record):
    """Write the result record atomically, never raising.

    This runs on the way out even when the run itself raised, and it must not
    replace that exception with one of its own — a rank that failed has to
    report why it failed. The write goes to a sibling temp file and is renamed
    into place, so an interruption mid-write cannot leave a truncated JSON for
    the gate to choke on.
    """
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(record, f)
        os.replace(tmp, path)
    except Exception as exc:
        logger.error("could not write result record to %s: %s", path, exc)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one rank of the Cosmic AI agentic pipeline")
    parser.add_argument("--galaxies", type=int, default=None,
                        help="number of galaxies (default: world_size, one per rank)")
    parser.add_argument("--dimensions", type=int,
                        default=int(os.environ.get("BEDROCK_EMBEDDING_DIMENSIONS", 256)))
    parser.add_argument("--max-chars", type=int, default=None,
                        help="truncate each prompt (default: no truncation)")
    parser.add_argument("--live", action="store_true",
                        help="use real Bedrock/Redis/DynamoDB — costs money on every rank")
    parser.add_argument("--blocking", action="store_true",
                        help="use FMI blocking mode instead of the default non-blocking one")
    parser.add_argument("--result-path", default=os.environ.get("RESULT_PATH"))
    parser.add_argument("--real-inference", action="store_true",
                        help="run AstroMAE over this rank's shard instead of the fixture")
    parser.add_argument("--data-path", default=os.environ.get("ASTROMAE_DATA_PATH"),
                        help="SDSS .pt partition (env ASTROMAE_DATA_PATH)")
    parser.add_argument("--model-path", default=os.environ.get("ASTROMAE_MODEL_PATH"),
                        help="AstroMAE weights .pt (env ASTROMAE_MODEL_PATH)")
    parser.add_argument("--batch-size", type=int,
                        default=int(os.environ.get("INFERENCE_BATCH_SIZE", 32)))
    parser.add_argument("--device", default=os.environ.get("INFERENCE_DEVICE", "cpu"),
                        help="torch device for inference (env INFERENCE_DEVICE)")
    parser.add_argument("--backend",
                        default=os.environ.get("EXECUTION_BACKEND", "armada"),
                        choices=[b.value for b in ExecutionBackend],
                        help="armada (collectives) or langchain (native invoke, Redis-shared context)")
    parser.add_argument("--epoch-batch-size", type=int,
                        default=int(os.environ.get("EPOCH_BATCH_SIZE", 4)),
                        help="galaxies per epoch; identical across arms (env EPOCH_BATCH_SIZE)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    comm_name = os.environ["COMM_NAME"]
    n_galaxies = args.galaxies or world_size
    workflow_id = f"cosmic_local_{comm_name}"

    backend = ExecutionBackend(args.backend)
    os.environ["CONTEXT_BACKEND"] = context_store_for(backend)

    services = live_services(args.dimensions) if args.live else mock_services(args.dimensions)
    metrics = RunMetrics()

    def build_for(rank_id):
        """The workflow wired for one rank identity.

        Built twice on Arm A: once with the requested rank to derive the peer
        map, which the channel needs before a bridge exists, and again with the
        rank the channel actually assigned. FMI hands out ranks by Redis INCR,
        so the two can differ, and an envelope stamped with the requested rank
        would misattribute which rank originated a context. The peer map does
        not depend on rank identity, only on the patterns the plan compiles to.
        """
        return build_cosmic_workflow(*services, workflow_id=workflow_id,
                                     dimensions=args.dimensions, max_chars=args.max_chars,
                                     metrics=metrics, rank=rank_id)

    establish_s = 0.0
    bridge = None
    if backend is ExecutionBackend.Armada:
        # Derived before the bridge exists: the channel establishes its
        # connections while the communicator is built, so the topology has to
        # be known first.
        peers = required_peer_map(build_for(rank), world_size)
        t0 = time.perf_counter()
        bridge = FMIBridge(
            world_size=world_size, rank=rank, channel_type="direct-redis",
            listen_port=int(os.environ.get("FMI_LISTEN_PORT", 10000)),
            redis_host=os.environ.get("REDIS_HOST", ""),
            redis_port=int(os.environ.get("REDIS_PORT", 6379)),
            comm_name=comm_name, maxtimeout=int(os.environ.get("FMI_MAX_TIMEOUT", 60000)),
            nonblocking=not args.blocking,
            advertise_host=os.environ.get("ADVERTISE_HOST", ""),
            required_peers=peers,
        )
        establish_s = time.perf_counter() - t0
        true_rank = bridge.rank
        channel = bridge.channel_type
    else:
        true_rank = rank
        channel = "none"

    seq = build_for(true_rank)

    root = 0
    logger.info("rank %d/%d ready in %.2fs (backend=%s, channel=%s, store=%s)",
                true_rank, world_size, establish_s, backend.value, channel,
                os.environ["CONTEXT_BACKEND"])

    inference_s = 0.0
    if args.real_inference:
        if not args.data_path or not args.model_path:
            raise SystemExit(
                "--real-inference needs --data-path and --model-path "
                "(or ASTROMAE_DATA_PATH / ASTROMAE_MODEL_PATH)"
            )
        t_inf = time.perf_counter()
        predictions, true_redshifts, magnitudes, _, (start, stop) = astromae_inference(
            args.data_path, args.model_path, world_size, true_rank,
            args.batch_size, args.device, n_galaxies=args.galaxies,
        )
        inference_s = time.perf_counter() - t_inf
    else:
        all_predictions, all_true, all_magnitudes = astromae_fixture(n_galaxies)
        start, stop = shard_bounds(n_galaxies, world_size, true_rank)
        predictions = all_predictions[start:stop]
        true_redshifts = all_true[start:stop]
        magnitudes = all_magnitudes[start:stop]

    placement = InputPlacement.PreDistributed

    n_local = len(predictions)
    prompts = generate_tasks_from_results(predictions, true_redshifts, magnitudes,
                                          max_tasks=n_local, seed=42)[:n_local]

    schema_in = seq.operators[0].schema_in

    def raw_table(texts):
        return pa.table({"raw_text": list(texts)}, schema=schema_in)

    shards = [raw_table(prompts[start:stop])
              for start, stop in plan_epochs(n_local, args.epoch_batch_size)]

    record = {"rank": true_rank, "world_size": world_size, "backend": backend.value,
              "context_backend": os.environ["CONTEXT_BACKEND"],
              "establish_s": round(establish_s, 4),
              "inference_s": round(inference_s, 4), "galaxies": n_local,
              "shard": [start, stop],
              "epochs": len(shards), "epoch_batch_size": args.epoch_batch_size,
              "placement": placement.value, "live": args.live}
    if args.real_inference:
        record["device"] = args.device

    results = []
    t1 = time.perf_counter()
    try:
        results = run_epochs(seq, shards, backend, bridge=bridge,
                             ctx=bridge._ctx if bridge is not None else None, root=root)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["run_s"] = round(time.perf_counter() - t1, 4)
        record.update(metrics.summary())
        record["acks_visible"] = sum(
            1
            for result in results
            if hasattr(result, "column_names") and "ack" in result.column_names
            for ack in result.column("ack").to_pylist()
            if ack)
        logger.info("rank %d done in %.2fs: %s", true_rank, record["run_s"], record)
        if args.result_path:
            write_record(args.result_path, record)

    if bridge is not None:
        if bridge.available:
            bridge.barrier()
        bridge.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())