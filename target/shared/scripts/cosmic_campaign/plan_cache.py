"""A plan cache shared between agents: off, exact reuse, or structural reuse."""

import base64
import json
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pyarrow as pa

from cosmic_campaign.grid import Configuration, RunSlot, num_workers, result_path
from cosmic_campaign.planner import PlanOutcome
from cosmic_campaign.requests import parse_parameters

PATH_MARKER = "/result-partition-"

ENTRY_SCHEMA = pa.schema([("request_text", pa.string()), ("embedding_b64", pa.string()),
                          ("plans_json", pa.string()), ("origin_rank", pa.int32())])
NO_ENTRY_RANK = -1


class CacheMode(Enum):
    OFF = "off"
    EXACT = "exact"
    STRUCTURAL = "structural"


@dataclass
class CacheEntry:
    request_text: str
    embedding: np.ndarray
    plans: list
    origin_rank: int


class PlanCache:
    def __init__(self, mode, similarity_threshold, settings, arm):
        self.mode, self.threshold, self.settings, self.arm = mode, similarity_threshold, settings, arm
        self.entries = []

    def __len__(self):
        return len(self.entries)

    def add(self, entry):
        self.entries.append(entry)

    def _nearest(self, embedding):
        best, best_score = None, float("-inf")
        for entry in self.entries:
            denom = np.linalg.norm(entry.embedding) * np.linalg.norm(embedding)
            score = float(np.dot(entry.embedding, embedding) / denom) if denom else -1.0
            if score > best_score:
                best, best_score = entry, score
        return best, best_score

    def _structural(self, text, cached_plans):
        """Keep the cached plan's fixed fields; derive the request's parameters and paths from the request."""
        if not cached_plans:
            return None
        try:
            p = parse_parameters(text)
        except ValueError:
            return None
        workers = num_workers(p["data_gb"], p["partition_mb"])
        config = Configuration(p["series"], p["partition_mb"], p["data_gb"], p["batch_size"], workers)
        derived = result_path("", self.arm, RunSlot(config, p["phase"], p["run_index"]))
        template = cached_plans[0]
        stem = str(template.get("result_path", "")).split(PATH_MARKER)[0]
        return [{**template, "file_limit": str(workers), "world_size": workers, "batch_size": p["batch_size"],
                 "data_prefix": f"{p['partition_mb']}MB",
                 "result_path": stem + derived[derived.index(PATH_MARKER):]}]

    def lookup(self, text, embedding):
        """Returns (outcome or None on a miss, match details: similarity and the nearest entry)."""
        if self.mode is CacheMode.OFF or not self.entries:
            return None, {}
        nearest, score = self._nearest(embedding)
        match = {"similarity": round(score, 6), "matched_request": nearest.request_text,
                 "matched_origin_rank": nearest.origin_rank}
        if score < self.threshold:
            return None, match
        if self.mode is CacheMode.EXACT:
            return PlanOutcome([dict(p) for p in nearest.plans], "cache_exact", 0.0, 0, 0, None), match
        plans = self._structural(text, nearest.plans)
        if plans is None:
            return None, match
        return PlanOutcome(plans, "cache_structural", 0.0, 0, 0, None), match

    @staticmethod
    def to_arrow(entries):
        """Flat columns only, and never empty: pycylon's AllGather drops list columns and crashes on empty tables."""
        if not entries:
            return pa.table({"request_text": [""], "embedding_b64": [""], "plans_json": ["[]"],
                             "origin_rank": [NO_ENTRY_RANK]}, schema=ENTRY_SCHEMA)
        return pa.table({
            "request_text": [e.request_text for e in entries],
            "embedding_b64": [base64.b64encode(np.asarray(e.embedding, dtype=np.float32).tobytes()).decode()
                              for e in entries],
            "plans_json": [json.dumps(e.plans) for e in entries],
            "origin_rank": [e.origin_rank for e in entries],
        }, schema=ENTRY_SCHEMA)

    @staticmethod
    def entries_from_arrow(table):
        return [CacheEntry(r["request_text"],
                           np.frombuffer(base64.b64decode(r["embedding_b64"]), dtype=np.float32).copy(),
                           json.loads(r["plans_json"]), r["origin_rank"])
                for r in table.to_pylist() if r["origin_rank"] != NO_ENTRY_RANK]
