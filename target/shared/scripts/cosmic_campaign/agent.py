"""One planning agent: plans its share of requests in rounds and shares new cache entries
with every other agent through one AllGather per round."""

import json
import math
import os
import sys
import time

from cosmic_campaign.plan_cache import CacheEntry, CacheMode, PlanCache
from cosmic_campaign.plan_check import mismatched_fields, validate_plan
from cosmic_campaign.planner import PlanOutcome


def assign_requests(requests, rank, world_size):
    return requests[rank::world_size]


def rounds_needed(n_requests, world_size):
    return math.ceil(n_requests / world_size)


def _record(request, outcome, rank, round_index, planning_ms, embedding_cost, match):
    mismatches = ([mismatched_fields(p, r) + validate_plan(p) for p, r in zip(outcome.plans, request.references)]
                  if len(outcome.plans) == len(request.references) else [["plan count"]])
    return {"request_id": request.request_id, "rank": rank, "round": round_index, "source": outcome.source,
            "latency_ms": outcome.latency_ms, "planning_ms": planning_ms,
            "input_tokens": outcome.input_tokens, "output_tokens": outcome.output_tokens,
            **embedding_cost, "similarity": match.get("similarity"),
            "matched_request": match.get("matched_request"),
            "matched_origin_rank": match.get("matched_origin_rank"),
            "error": outcome.error, "plans": outcome.plans, "references": list(request.references),
            "mismatches": mismatches}


def _embed(request, cache, embed):
    if cache.mode is CacheMode.OFF:
        return None, {"embed_ms": 0.0, "embed_tokens": 0, "embed_cache_hit": False}
    start = time.perf_counter()
    embedding, meta = embed(request.text)
    return embedding, {"embed_ms": (time.perf_counter() - start) * 1000,
                       "embed_tokens": meta.get("token_count", 0),
                       "embed_cache_hit": bool(meta.get("cache_hit", False))}


def _plan(request, planner):
    start = time.perf_counter()
    try:
        return planner.plan(request)
    except Exception as exc:
        return PlanOutcome([], planner.source, (time.perf_counter() - start) * 1000, 0, 0,
                           f"{type(exc).__name__}: {exc}")


def run_agent(rank, world_size, requests, planner, cache, embed, share, manifest_path, summary_path):
    mine = assign_requests(requests, rank, world_size)
    rows, share_ms, started = [], [], time.perf_counter()
    with open(manifest_path, "w") as manifest:
        for round_index in range(rounds_needed(len(requests), world_size)):
            new_entries, row = [], None
            if round_index < len(mine):
                request = mine[round_index]
                start = time.perf_counter()
                embedding, embedding_cost = _embed(request, cache, embed)
                outcome, match = cache.lookup(request.text, embedding) if embedding is not None else (None, {})
                outcome = outcome or _plan(request, planner)
                row = _record(request, outcome, rank, round_index, (time.perf_counter() - start) * 1000,
                              embedding_cost, match)
                if outcome.source.startswith("llm") and outcome.plans and embedding is not None:
                    new_entries.append(CacheEntry(request.text, embedding, outcome.plans, rank))
            start = time.perf_counter()
            for table in share(PlanCache.to_arrow(new_entries)):
                for entry in PlanCache.entries_from_arrow(table):
                    cache.add(entry)
            share_ms.append((time.perf_counter() - start) * 1000)
            if row is not None:
                row["share_ms"] = share_ms[-1]
                manifest.write(json.dumps(row) + "\n")
                manifest.flush()
                rows.append(row)
    with open(summary_path, "w") as summary:
        json.dump({"rank": rank, "wall_ms": (time.perf_counter() - started) * 1000, "share_ms": share_ms}, summary)
    return rows


def fmi_share(bridge, world_size):
    if not bridge.available:
        if world_size > 1:
            raise RuntimeError(f"FMI communicator unavailable for {world_size} agents; they would not share plans")
        return lambda table: [table]
    from pycylon import Table

    def share(table):
        return [t.to_arrow() for t in bridge.allgather(Table.from_arrow(bridge.context, table))]

    return share


def _planner(args, settings, arm):
    from chain.executor import ChainExecutor
    from cosmic_campaign.planner import LLMPlanner, ToolPlanner
    from cost.bedrock_pricing import BedrockConfig

    if args.planner == "llm":
        return LLMPlanner(ChainExecutor(), settings, arm)
    from langchain_aws import ChatBedrockConverse

    config = BedrockConfig.resolve()
    model = ChatBedrockConverse(model=config.llm_model_id, region_name=config.region, temperature=0.0)
    return ToolPlanner(model, settings, arm, max_steps=args.max_tool_steps)


def main(argv=None):
    import argparse

    from communicator.fmi_bridge import FMIBridge
    from context.embedding import EmbeddingService
    from cosmic_campaign.grid import CampaignSettings
    from cosmic_campaign.requests import Request

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--requests", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache-mode", choices=[m.value for m in CacheMode], required=True)
    p.add_argument("--similarity-threshold", type=float, required=True)
    p.add_argument("--planner", choices=["llm", "llm-tool"], default="llm")
    p.add_argument("--max-tool-steps", type=int, default=4)
    args = p.parse_args(argv)
    spec = json.load(open(args.requests))
    settings = CampaignSettings(**spec["settings"])
    requests = [Request(r["request_id"], r["kind"], r["text"], tuple(), tuple(r["references"]))
                for r in spec["requests"]]
    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    bridge = FMIBridge(
        world_size=world_size, rank=rank,
        channel_type=os.environ["FMI_CHANNEL_TYPE"],
        comm_name=os.environ["COMM_NAME"],
        listen_port=int(os.environ["FMI_LISTEN_PORT"]),
        redis_host=os.environ["REDIS_HOST"],
        redis_port=int(os.environ["REDIS_PORT"]),
        advertise_host=os.environ.get("ADVERTISE_HOST", ""),
    )
    embedder = EmbeddingService(cache_embeddings=False)
    cache = PlanCache(CacheMode(args.cache_mode), args.similarity_threshold, settings, spec["arm"])
    try:
        run_agent(rank, world_size, requests, _planner(args, settings, spec["arm"]), cache, embedder.embed,
                  fmi_share(bridge, world_size), os.path.join(args.out_dir, f"plans_{rank}.jsonl"),
                  os.path.join(args.out_dir, f"agent_{rank}.json"))
    finally:
        bridge.finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())