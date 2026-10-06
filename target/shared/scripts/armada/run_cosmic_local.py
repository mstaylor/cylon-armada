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
workflow and executes it under one of five backends (ExecutionBackend):
armada and ray-cylon open an FMIBridge over the direct-redis channel, with
ray-cylon building it through CylonRayActor inside a Ray cluster; ray-native
shares contexts through the Ray object store with one end-of-run barrier;
langchain shares through Redis; isolated shares nothing.

Ray arms record ray_cluster_s (forming the cluster) apart from establish_s
(the data plane's own setup), and ray-native records teardown_barrier_s (the
end-of-run barrier and late drain) apart from run_s.

Services are mocked by default: --live switches to real Bedrock/Redis/DynamoDB
and therefore costs money on every rank (one embedding call and one LLM call per
galaxy), so it is opt-in rather than the default.

Environment (set by the launcher): RANK, WORLD_SIZE, COMM_NAME, REDIS_HOST,
REDIS_PORT, FMI_LISTEN_PORT. Ray arms also read RAY_PORT,
RAY_RENDEZVOUS_TIMEOUT_S, RAY_CLUSTER_TIMEOUT_S, RAY_BARRIER_TIMEOUT_S and
RAY_NAMESPACE.

Rank identity differs by arm. Under the armada and ray-cylon backends the rank
is the one the FMI channel negotiates through Redis INCR (bridge.rank), which
can differ from the RANK the launcher requested; under the other backends there
is no channel to negotiate through, so the rank is RANK itself. Both are used to pick
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
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pyarrow as pa

from armada.cosmic_workflow import build_cosmic_workflow
from armada.epochs import epoch_count, plan_epochs
from armada.executor import ArmadaExecutor, InputPlacement, required_peer_map
from armada.ray_native_executor import RayNativeExecutor
from armada.reuse_policy import (
    PHOTOMETRY_BANDS,
    is_photometry_key,
    photometry_reuse_key_for,
    redshift_validator,
)
from armada.run_metrics import RunMetrics
from communicator.fmi_bridge import FMIBridge
from cosmic_ai.task_generator import (
    assert_default_photometry_template,
    bands_for,
    generate_tasks_from_results,
    template_for,
)
from ray_arm.cluster import join_by_discovery, start_head, wait_for_nodes
from ray_arm.cylon_on_ray import CylonRayActor

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


def outlier_threshold_for():
    """Fixed residual above which a galaxy gets the outlier prompt, or None.

    Left unset, the generator derives the threshold from whichever galaxies are
    passed to it — which in a sharded run is this rank's slice, so the same
    galaxy gets a different prompt at different world sizes and the workload
    moves with the independent variable. Any sweep that varies N must pin this
    to a value derived once from the whole population, so a prompt is a
    property of its galaxy alone.
    """
    from armada.reuse_policy import _validated

    raw = os.environ.get("OUTLIER_RESIDUAL_THRESHOLD")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"OUTLIER_RESIDUAL_THRESHOLD={raw!r} is not a number; it is a "
            f"redshift residual such as 0.0295"
        ) from None
    return _validated("OUTLIER_RESIDUAL_THRESHOLD", raw, value)


def collective_epochs(n_items, world_size, batch_size):
    """How many epochs EVERY rank must run, not just this one.

    Each epoch fires one AllGather, which every rank has to reach. Shards are
    not all the same size — at 1253 galaxies and N=8, five ranks hold 157 and
    three hold 156, so with batch 4 some ranks plan 40 epochs and others 39.
    A rank that finishes early simply stops calling the collective while its
    peers are still waiting in it, and the run hangs until the FMI timeout.

    So the epoch count is a property of the largest shard, and ranks with less
    work contribute empty tables in their trailing epochs. The executor already
    handles zero-row contributions; what it cannot handle is a participant that
    never arrives.
    """
    largest = max(stop - start for start, stop in
                  (shard_bounds(n_items, world_size, rank) for rank in range(world_size)))
    return epoch_count(largest, batch_size)


def reuse_keys_for(predictions, true_redshifts, residuals, magnitudes,
                   outlier_threshold, index_offset, n_local):
    """Per-row reuse_key, template-aware. See
    docs/Reuse_Gate_Validity_Findings_2026-09-28.md for the design and
    measurements behind each template's key choice.
    """
    keys = []
    for idx in range(n_local):
        template = template_for(idx, float(residuals[idx]), outlier_threshold, index_offset)
        if template == "redshift_analysis":
            keys.append(_continuous_key(float(predictions[idx]) - float(true_redshifts[idx])))
        elif template == "outlier_analysis":
            keys.append(_continuous_key(float(predictions[idx])))
        elif template == "photometry_classification":
            keys.append(photometry_reuse_key_for(magnitudes[idx]))
        else:
            keys.append(None)
    return keys


def _continuous_key(value):
    """None if value falls in photometry_classification's reserved key
    range — see docs/Reuse_Gate_Validity_Findings_2026-09-28.md.
    """
    if is_photometry_key(value):
        return None
    return value


def photometry_features_for(magnitudes, predictions, true_redshifts, residuals,
                            outlier_threshold, index_offset, n_local):
    """Per-row photometry_features passthrough: this row's own 5 magnitude
    values for a valid photometry_classification row, else None. See
    docs/Reuse_Gate_Validity_Findings_2026-09-28.md.
    """
    features = []
    for idx in range(n_local):
        template = template_for(idx, float(residuals[idx]), outlier_threshold, index_offset)
        if template != "photometry_classification":
            features.append(None)
        elif photometry_reuse_key_for(magnitudes[idx]) is None:
            features.append(None)
        else:
            features.append([float(v) for v in magnitudes[idx]])
    return features


def assert_photometry_band_order(config_path=None):
    """Refuse to run if the resolved band order does not match
    reuse_policy.PHOTOMETRY_BANDS."""
    resolved = tuple(bands_for(config_path=config_path))
    if resolved != PHOTOMETRY_BANDS:
        raise ValueError(
            f"resolved band order {resolved!r} does not match "
            f"reuse_policy.PHOTOMETRY_BANDS {PHOTOMETRY_BANDS!r}"
        )


def assert_photometry_config_is_safe(config_path=None):
    """Both preconditions a gated photometry run depends on: band order and
    template text both match the built-in default."""
    assert_photometry_band_order(config_path=config_path)
    assert_default_photometry_template(config_path=config_path)


def workload_hash(start, prompts, reuse_keys):
    """Identity of the workload this rank actually ran.

    A scaling curve is only interpretable if a galaxy's prompt and key are the
    same at every world size, and the generator has two routes to making them
    otherwise — a residual percentile and an index parity, both computed over
    whichever rows were handed in. Recording the hash makes that a fact the
    results carry rather than a property the runner is trusted to have
    preserved.

    This is a per-shard digest and it is NOT partition-independent: shard
    boundaries move with N, so concatenating per-rank hashes gives a different
    string at a different world size even over identical galaxies. It proves
    corresponding shards match at a FIXED N. Proving the galaxies are frozen
    ACROSS N needs a digest over the whole population in global index order,
    which is what the readiness check computes.
    """
    import hashlib

    digest = hashlib.sha256()
    for offset, (prompt, key) in enumerate(zip(prompts, reuse_keys)):
        digest.update(f"{start + offset}\x1f{prompt}\x1f{key!r}\x1e".encode())
    return digest.hexdigest()[:16]


def enforce_isolation(backend):
    """Make the isolated arm's isolation a property of the arm, not of a flag.

    A cylon-backed store still snapshots its whole ContextTable to the shared
    Redis key `context_table:{workflow_id}` on every store unless the snapshot
    is off — so an isolated rank would write into a store other ranks also
    write to, and the control would not be a control. Until now that was
    prevented only by the Fargate driver setting CONTEXT_TABLE_SNAPSHOT=0, and
    a local `--backend isolated --live` run had no such protection.

    An environment that explicitly asks for the snapshot on this arm is refused
    rather than overridden, the same way a conflicting CONTEXT_BACKEND is.
    """
    if backend is not ExecutionBackend.Isolated:
        return

    configured = os.environ.get("CONTEXT_TABLE_SNAPSHOT")
    if configured is not None and configured.strip().lower() in ("1", "true", "yes", "on"):
        raise ValueError(
            f"CONTEXT_TABLE_SNAPSHOT={configured!r} conflicts with the isolated arm: "
            f"the snapshot writes this rank's whole table to a Redis key every rank "
            f"shares, so the no-sharing control would share"
        )
    os.environ["CONTEXT_TABLE_SNAPSHOT"] = "0"


def pricing_for():
    """Bedrock pricing used to turn token counts into dollars.

    Resolution is the module's own chain: config file, then the AWS Pricing
    API, then static defaults. A run records which source it got, because a
    cost figure is only comparable across runs if the price table was the
    same, and the static fallback is not the live price.
    """
    from cost.bedrock_pricing import BedrockPricing

    return BedrockPricing.resolve(region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


def reuse_policy_for():
    """The reuse validity policy for this run, or None to leave reuse ungated.

    Gating is opt-in at the run level so an ungated control stays possible and
    directly comparable with every measurement taken before the gate existed.
    Set REUSE_KEY_TOLERANCE to enable it.
    """
    if os.environ.get("REUSE_KEY_TOLERANCE") is None:
        return None
    return redshift_validator()


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

    config = BedrockConfig.resolve(payload={"embedding_dimensions": dimensions})
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

    Isolated shares nothing: a rank sees only the contexts it created itself.
    It is the controlled analogue of the shipped Cosmic AI design, which runs
    as a Step Functions Distributed Map at 3000-way concurrency with no
    inter-worker communication of any kind. It exists to measure what that
    isolation costs once workers could have reused each other's work, so its
    ranks must never reach a collective or a store another rank writes to.
    """

    Armada = "armada"
    LangChain = "langchain"
    Isolated = "isolated"
    RayNative = "ray-native"
    RayCylon = "ray-cylon"


_CONTEXT_STORE = {
    ExecutionBackend.Armada: "cylon",
    ExecutionBackend.LangChain: "redis",
    ExecutionBackend.Isolated: "cylon",
    ExecutionBackend.RayNative: "plasma",
    ExecutionBackend.RayCylon: "cylon",
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


def _redis_client_for_rendezvous():
    import redis

    host = os.environ.get("REDIS_HOST", "")
    if not host:
        raise ValueError("REDIS_HOST must be set: the Ray arms publish and discover the head address through Redis")
    return redis.Redis(host=host, port=int(os.environ.get("REDIS_PORT", 6379)))


def form_ray_cluster(rank, world_size, comm_name, redis_client, ray_api):
    """Join this rank to the run's Ray cluster, rendezvousing through Redis.

    Rank 0 starts the head and publishes its address under comm_name; every
    other rank discovers that address and joins. Returns once the cluster
    reports world_size live nodes, so establish_s covers the whole formation.
    """
    if rank == 0:
        start_head(int(os.environ.get("RAY_PORT", 6380)), redis_client, comm_name)
    else:
        join_by_discovery(redis_client, comm_name,
                          timeout_s=float(os.environ.get("RAY_RENDEZVOUS_TIMEOUT_S", 300)))
    ray_api.init(address="auto", ignore_reinit_error=True)
    wait_for_nodes(world_size,
                   timeout_s=float(os.environ.get("RAY_CLUSTER_TIMEOUT_S", 300)),
                   nodes_fn=lambda: sum(1 for node in ray_api.nodes() if node.get("Alive")))


@dataclass(frozen=True)
class RayNativeHandles:
    """The actors one ray-native rank uses: its own shard, and the run's
    shared registry and end-of-run barrier."""

    shard_actor: object
    registry: object
    barrier: object


def ray_native_actors(rank, world_size, comm_name, ray_api):
    """This rank's ShardActor, pinned to its own node, plus the run's named
    registry and barrier.

    The registry and barrier are named per run so concurrent runs on one
    cluster never share them, placed in the RAY_NAMESPACE namespace because
    each rank is its own driver with its own anonymous namespace, and
    detached so they outlive rank 0's driver until release_ray_native_run
    kills them. Pinning keeps each rank's
    ray.put in its own node's object store; left to the scheduler, a rank's
    actor could land on another task's node and turn its local reads into
    network transfers. Returns only once the shard and registry actors are
    running: .remote() returns before the actor process exists, so without
    the wait their startup would be charged to the first epoch's run_s
    instead of establish_s.
    """
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from ray_arm.native import ContextRegistry, RunBarrier, ShardActor

    namespace = os.environ.get("RAY_NAMESPACE", "cylon-armada")
    registry = ContextRegistry.options(
        name=f"cosmic_registry_{comm_name}", namespace=namespace,
        get_if_exists=True, lifetime="detached"
    ).remote()
    barrier = RunBarrier.options(
        name=f"cosmic_barrier_{comm_name}", namespace=namespace,
        get_if_exists=True, lifetime="detached"
    ).remote(world_size)
    node_id = ray_api.get_runtime_context().get_node_id()
    shard_actor = ShardActor.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
    ).remote(rank, world_size, registry)
    ray_api.get([shard_actor.watermark.remote(), registry.count.remote()])
    return RayNativeHandles(shard_actor=shard_actor, registry=registry, barrier=barrier)


def release_ray_native_run(handles, rank, world_size, ray_api, timeout_s):
    """Kill the run's named actors once no rank can still need them.

    Called after RayNativeExecutor.finish, so every rank is already past its
    last read. Rank 0 alone cleans up, after every other rank has departed;
    the others only depart. A departure whose reply is lost because rank 0
    killed the barrier right after counting it is the expected end state,
    not a failure.
    """
    if rank == 0:
        ray_api.get(handles.barrier.wait_departed.remote(world_size - 1), timeout=timeout_s)
        ray_api.kill(handles.registry)
        ray_api.kill(handles.barrier)
        return
    try:
        ray_api.get(handles.barrier.depart.remote(), timeout=timeout_s)
    except ray_api.exceptions.RayActorError:
        logger.info("rank %d departed as the run's barrier was released", rank)


def run_epochs(seq, shards, backend, bridge=None, ctx=None, root=0, shard_actor=None):
    """Execute one epoch per shard, returning each epoch's result table.

    Under Armada and ray-cylon each epoch is one executor pass ending in
    MemoryUpsert's AllGather, moved over FMI either way — ray-cylon differs
    only in how the bridge was constructed, never in how it runs. Under
    ray-native each epoch runs locally except MemoryUpsert, which publishes
    to and reads from the object store instead. Under LangChain/Isolated
    nothing crosses ranks because the store itself is shared.
    """
    results = []
    for shard in shards:
        if backend in (ExecutionBackend.LangChain, ExecutionBackend.Isolated):
            results.append(seq.invoke(shard))
        elif backend in (ExecutionBackend.Armada, ExecutionBackend.RayCylon):
            results.append(ArmadaExecutor(bridge).run(
                seq, input_tables=shard, ctx=ctx, root=root,
                placement=InputPlacement.PreDistributed))
        elif backend is ExecutionBackend.RayNative:
            results.append(RayNativeExecutor(shard_actor).run(seq, shard))
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
                        help="armada (Cylon collectives over FMI), langchain (native invoke, "
                             "Redis-shared context), isolated (no sharing), ray-native (Ray "
                             "object store), or ray-cylon (Ray cluster, Cylon collectives "
                             "over FMI); env EXECUTION_BACKEND")
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
    enforce_isolation(backend)

    services = live_services(args.dimensions) if args.live else mock_services(args.dimensions)
    metrics = RunMetrics()
    pricing = pricing_for()

    def build_for(rank_id):
        """The workflow wired for one rank identity.

        Built twice on Arm A: once with the requested rank to derive the peer
        map, which the channel needs before a bridge exists, and again with the
        rank the channel actually assigned. FMI hands out ranks by Redis INCR,
        so the two can differ, and an envelope stamped with the requested rank
        would misattribute which rank originated a context. The peer map does
        not depend on rank identity, only on the patterns the plan compiles to.

        The reuse key is not passed here: it rides with the rows as a
        passthrough column, because this workflow's operators are invoked once
        per epoch and a positional list would restamp the first epoch's keys
        onto every later one.
        """
        return build_cosmic_workflow(*services, workflow_id=workflow_id,
                                     dimensions=args.dimensions, max_chars=args.max_chars,
                                     metrics=metrics, rank=rank_id,
                                     reuse_validator=reuse_policy_for(),
                                     pricing=pricing)

    establish_s = 0.0
    ray_cluster_s = None
    bridge = None
    ray_native = None
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
    elif backend in (ExecutionBackend.RayNative, ExecutionBackend.RayCylon):
        import ray

        t_cluster = time.perf_counter()
        form_ray_cluster(rank, world_size, comm_name, _redis_client_for_rendezvous(), ray)
        ray_cluster_s = time.perf_counter() - t_cluster
        t0 = time.perf_counter()
        true_rank = rank
        if backend is ExecutionBackend.RayNative:
            ray_native = ray_native_actors(true_rank, world_size, comm_name, ray)
            channel = "ray-plasma"
        else:
            actor = CylonRayActor(rank=rank, world_size=world_size, comm_name=comm_name,
                                  required_peers=required_peer_map(build_for(rank), world_size),
                                  nonblocking=not args.blocking)
            actor.start_env()
            bridge = actor.bridge
            true_rank = actor.rank
            channel = bridge.channel_type
        establish_s = time.perf_counter() - t0
    else:
        true_rank = rank
        channel = "none"

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
    residuals = np.abs(predictions - true_redshifts)
    # Resolved explicitly (not left as None) so generate_tasks_from_results'
    # own template choice and reuse_keys_for's per-row template lookup agree
    # on the exact same threshold — passing None to both independently would
    # let each derive its own np.percentile call and silently drift apart.
    outlier_threshold = outlier_threshold_for()
    if outlier_threshold is None:
        outlier_threshold = float(np.percentile(residuals, 90))
    gated = reuse_policy_for() is not None
    if gated:
        assert_photometry_config_is_safe()
    prompts = generate_tasks_from_results(predictions, true_redshifts, magnitudes,
                                          max_tasks=n_local, seed=42,
                                          outlier_threshold=outlier_threshold,
                                          index_offset=start)[:n_local]
    reuse_keys = reuse_keys_for(predictions, true_redshifts, residuals, magnitudes,
                                outlier_threshold, start, n_local)
    photometry_features = photometry_features_for(magnitudes, predictions, true_redshifts,
                                                   residuals, outlier_threshold, start, n_local)

    seq = build_for(true_rank)
    photometry_features_type = pa.list_(pa.float64(), 5)

    def raw_table(texts, keys, features):
        columns = {"raw_text": pa.array(list(texts), type=pa.large_utf8())}
        if gated:
            columns["reuse_key"] = pa.array(list(keys), type=pa.float64())
            columns["photometry_features"] = pa.array(list(features), type=photometry_features_type)
        return pa.table(columns)

    shards = [raw_table(prompts[lo:hi], reuse_keys[lo:hi], photometry_features[lo:hi])
              for lo, hi in plan_epochs(n_local, args.epoch_batch_size)]
    shards += [raw_table([], [], [])
               for _ in range(collective_epochs(n_galaxies, world_size,
                                                args.epoch_batch_size) - len(shards))]

    record = {"rank": true_rank, "world_size": world_size, "backend": backend.value,
              "context_backend": os.environ["CONTEXT_BACKEND"],
              "establish_s": round(establish_s, 4),
              "inference_s": round(inference_s, 4), "galaxies": n_local,
              "shard": [start, stop],
              "epochs": len(shards), "epoch_batch_size": args.epoch_batch_size,
              "placement": placement.value, "live": args.live,
              "workload_hash": workload_hash(start, prompts, reuse_keys)}
    if args.real_inference:
        record["device"] = args.device

    record["pricing_source"] = pricing.source
    if ray_cluster_s is not None:
        record["ray_cluster_s"] = round(ray_cluster_s, 4)
    ray_barrier_timeout_s = float(os.environ.get("RAY_BARRIER_TIMEOUT_S", 300))

    results = []
    t1 = time.perf_counter()
    run_end = None
    try:
        results = run_epochs(seq, shards, backend, bridge=bridge,
                             ctx=bridge._ctx if bridge is not None else None, root=root,
                             shard_actor=ray_native.shard_actor if ray_native else None)
        run_end = time.perf_counter()
        if ray_native is not None:
            late = RayNativeExecutor(ray_native.shard_actor).finish(
                seq, ray_native.barrier, ray_barrier_timeout_s)
            record["teardown_barrier_s"] = round(time.perf_counter() - run_end, 4)
            if late is not None:
                results.append(late)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["run_s"] = round((run_end or time.perf_counter()) - t1, 4)
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
    if backend in (ExecutionBackend.RayNative, ExecutionBackend.RayCylon):
        import ray

        if ray_native is not None:
            release_ray_native_run(ray_native, true_rank, world_size, ray, ray_barrier_timeout_s)
        ray.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())