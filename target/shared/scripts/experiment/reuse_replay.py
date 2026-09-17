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

"""Offline replay of the context-reuse mechanism.

Answers the question the Experiment E headline claim rests on — does isolating
workers destroy reuse — without running a rank, calling an LLM, or paying for
Fargate. The reuse decision is entirely `find_similar` plus a threshold over
embeddings, so it is exactly simulatable from one embedding matrix.

Modelled faithfully:
  - cosine similarity against a threshold, matching ContextRouter
  - contiguous sharding, via the runner's own shard_bounds
  - epoch batching, via the runner's own plan_epochs
  - within-epoch blindness: Retrieve runs over a whole epoch's rows before
    MemoryUpsert stores any of them, so a galaxy cannot reuse a context created
    earlier in its own epoch
  - a cache hit stores nothing, so only misses enter the store

Not modelled: time. No transport cost, no contention, no barrier wait. This
answers "does isolation cost reuse", never "which arm is faster".

Arms collapse to two topologies, which is the honest granularity: `isolated`
(each rank sees only its own contexts — the embarrassingly-parallel design) and
`shared` (every rank sees every rank's, whether by collective or by a
centralized store). Separating Armada from LangChain is a timing question this
cannot answer.

Run: python -m experiment.reuse_replay --help
"""

import argparse
import json

import numpy as np

from armada.epochs import plan_epochs
from armada.run_cosmic_local import shard_bounds

ISOLATED = "isolated"
SHARED = "shared"
TOPOLOGIES = (ISOLATED, SHARED)


def normalize(matrix):
    """Unit-norm rows, so cosine similarity is a plain dot product.

    A zero row is left at zero rather than divided; its similarity to anything
    is then 0, the same answer ContextRouter's scalar fallback gives for a
    zero-norm vector.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


class RankReplay:
    """One rank's view: which contexts it can see, and what it hit."""

    def __init__(self, rank, shard, embeddings, threshold, top_k=5,
                 gate_values=None, gate_tolerance=None):
        self.rank = rank
        self.shard = shard
        self.threshold = threshold
        self.top_k = top_k
        self._embeddings = embeddings
        self._gate_values = gate_values
        self._gate_tolerance = gate_tolerance
        self._visible = []
        self.retrievals = 0
        self.cache_hits = 0
        self.llm_calls = 0
        self.gate_rejections = 0
        self.gate_errors = []

    @property
    def galaxies(self):
        start, stop = self.shard
        return stop - start

    def _is_hit(self, index):
        """Cosine above threshold, and — when gated — a reuse that is valid.

        Ungated this is the deployed behaviour: any stored context above the
        threshold is reused. Gated, the top_k candidates by cosine are filtered
        on how far the reused context's gate value sits from the query's, which
        is how a domain constraint ("the analysis must be about a galaxy at
        essentially this redshift") is expressed. The filter runs over the
        candidate list rather than the single best match, so a slightly worse
        cosine match that is valid still counts.
        """
        if not self._visible:
            return False

        candidates = np.asarray(self._visible)
        sims = self._embeddings[candidates] @ self._embeddings[index]
        above = sims >= self.threshold
        if not above.any():
            return False

        if self._gate_values is None or self._gate_tolerance is None:
            return True

        ranked = candidates[above][np.argsort(-sims[above])][: self.top_k]
        errors = np.abs(self._gate_values[ranked] - self._gate_values[index])
        if not (errors <= self._gate_tolerance).any():
            self.gate_rejections += 1
            return False

        self.gate_errors.append(float(errors.min()))
        return True

    def run_epoch(self, indices):
        """Retrieve for every row, and report which ones missed.

        Retrieval for the whole epoch runs against the store as it stood at
        epoch start, because the pipeline runs Retrieve over every row before
        MemoryUpsert stores any of them.
        """
        missed = []
        for index in indices:
            self.retrievals += 1
            if self._is_hit(index):
                self.cache_hits += 1
            else:
                self.llm_calls += 1
                missed.append(index)
        return missed

    def ingest(self, indices):
        self._visible.extend(indices)

    def summary(self):
        return {
            "rank": self.rank,
            "shard": list(self.shard),
            "galaxies": self.galaxies,
            "retrievals": self.retrievals,
            "cache_hits": self.cache_hits,
            "llm_calls": self.llm_calls,
            "reuse_rate": (self.cache_hits / self.retrievals) if self.retrievals else 0.0,
            "contexts_visible": len(self._visible),
            "gate_rejections": self.gate_rejections,
            "worst_gate_error": max(self.gate_errors) if self.gate_errors else 0.0,
        }


def replay(embeddings, world_size, topology, threshold=0.85, epoch_batch_size=4,
           order=None, top_k=5, gate_values=None, gate_tolerance=None):
    """Replay one (world_size, topology) point over the whole population.

    `order` permutes the catalogue before sharding, which is how a caller asks
    whether the result depends on partitioning: contiguous sharding of a
    catalogue whose neighbours are already similar flatters the isolated
    topology, and shuffling removes that advantage.
    """
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}, expected one of {TOPOLOGIES}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    embeddings = normalize(embeddings)
    n_items = embeddings.shape[0]
    order = np.arange(n_items) if order is None else np.asarray(order)

    gate_values = None if gate_values is None else np.asarray(gate_values, dtype=np.float64)
    ranks = [
        RankReplay(rank, shard_bounds(n_items, world_size, rank), embeddings, threshold,
                   top_k, gate_values, gate_tolerance)
        for rank in range(world_size)
    ]
    epochs_per_rank = [plan_epochs(r.galaxies, epoch_batch_size) for r in ranks]

    for epoch in range(max((len(e) for e in epochs_per_rank), default=0)):
        produced = {}
        for rank_replay, epochs in zip(ranks, epochs_per_rank):
            if epoch >= len(epochs):
                continue
            lo, hi = epochs[epoch]
            indices = [int(order[rank_replay.shard[0] + offset]) for offset in range(lo, hi)]
            produced[rank_replay.rank] = rank_replay.run_epoch(indices)

        everyones_misses = [i for misses in produced.values() for i in misses]
        for rank_replay in ranks:
            if topology == SHARED:
                rank_replay.ingest(everyones_misses)
            elif rank_replay.rank in produced:
                rank_replay.ingest(produced[rank_replay.rank])

    return [r.summary() for r in ranks]


def totals(rank_summaries):
    """Job-level totals, which is what a cost claim is made of."""
    galaxies = sum(r["galaxies"] for r in rank_summaries)
    llm_calls = sum(r["llm_calls"] for r in rank_summaries)
    hits = sum(r["cache_hits"] for r in rank_summaries)
    return {
        "world_size": len(rank_summaries),
        "galaxies": galaxies,
        "llm_calls": llm_calls,
        "cache_hits": hits,
        "reuse_rate": (hits / galaxies) if galaxies else 0.0,
    }


def sweep(embeddings, world_sizes, threshold=0.85, epoch_batch_size=4, order=None):
    """Both topologies across the world sizes, as job-level totals."""
    return {
        topology: [
            totals(replay(embeddings, n, topology, threshold, epoch_batch_size, order))
            for n in world_sizes
        ]
        for topology in TOPOLOGIES
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--embeddings", required=True,
                        help="path to a .npy embedding matrix (n_galaxies x D)")
    parser.add_argument("--world-sizes", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64],
                        help="must include 1 and 2 as baselines")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="reuse threshold; matches SIMILARITY_THRESHOLD")
    parser.add_argument("--epoch-batch-size", type=int, default=4)
    parser.add_argument("--shuffle-seed", type=int, default=None,
                        help="permute the catalogue before sharding, to test whether "
                             "the result depends on contiguous partitioning")
    parser.add_argument("--json", action="store_true", help="emit JSON rather than a table")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    embeddings = np.load(args.embeddings)

    order = None
    if args.shuffle_seed is not None:
        order = np.random.default_rng(args.shuffle_seed).permutation(embeddings.shape[0])

    result = sweep(embeddings, args.world_sizes, args.threshold,
                   args.epoch_batch_size, order)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"population={embeddings.shape[0]}  dim={embeddings.shape[1]}  "
          f"threshold={args.threshold}  epoch_batch={args.epoch_batch_size}  "
          f"order={'shuffled' if order is not None else 'catalogue'}")
    print()
    print(f"{'N':>5}{'isolated calls':>16}{'shared calls':>14}"
          f"{'isolated reuse':>16}{'shared reuse':>14}{'extra calls':>13}")
    print("-" * 78)
    for iso, shr in zip(result[ISOLATED], result[SHARED]):
        extra = iso["llm_calls"] - shr["llm_calls"]
        ratio = (iso["llm_calls"] / shr["llm_calls"]) if shr["llm_calls"] else float("inf")
        print(f"{iso['world_size']:>5}{iso['llm_calls']:>16}{shr['llm_calls']:>14}"
              f"{iso['reuse_rate']:>16.3f}{shr['reuse_rate']:>14.3f}"
              f"{extra:>10} ({ratio:.2f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())