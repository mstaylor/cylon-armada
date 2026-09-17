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

"""Correctness gate for an Experiment E paired run.

No timing from a paired run is believed until both arms are shown to have
done the same work. The gate reads the per-rank result records each arm
produced and checks, per arm, that every rank reported exactly once, that the
ranks' shards tile the galaxy population with no gap or overlap, that every
galaxy produced a stored record, and that nothing failed or was throttled —
then, across arms, that both covered the same population.

Response text is deliberately never compared. LLM output is nondeterministic
across arms, so a gate on text would fail every run for a reason that says
nothing about the data plane. A run that fails the gate is discarded and
repeated, never averaged in.

Each failure string names the arm, the rank, and the observed versus expected
value, so a gate failure is debuggable from the message alone.
"""

from dataclasses import dataclass, field


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)


def _check_arm(arm, records, failures):
    """Validate one arm's records; return its total galaxy coverage, or None.

    Every galaxy in a rank's shard must be accounted for, but a galaxy whose
    context was reused stores nothing — a hit deliberately writes no duplicate
    — so the invariant is records_written + cache_hits == galaxies rather than
    records_written alone. Both terms are recorded per rank on both arms, and
    written-plus-hits falling short of the shard is still a lost record.
    """
    if not records:
        failures.append(f"{arm}: no rank records")
        return None

    world_sizes = {r["world_size"] for r in records}
    if len(world_sizes) != 1:
        failures.append(f"{arm}: ranks disagree on world_size {sorted(world_sizes)}")
        return None
    world_size = world_sizes.pop()

    ranks = sorted(r["rank"] for r in records)
    expected = list(range(world_size))
    if ranks != expected:
        missing = sorted(set(expected) - set(ranks))
        duplicated = sorted({r for r in ranks if ranks.count(r) > 1})
        failures.append(
            f"{arm}: rank set mismatch for world_size {world_size} — "
            f"missing {missing}, duplicated {duplicated}, got {ranks}"
        )

    for r in records:
        start, stop = r["shard"]
        if not (0 <= start <= stop):
            failures.append(
                f"{arm} rank {r['rank']}: malformed shard {list(r['shard'])} — "
                f"expected 0 <= start <= stop"
            )

    cursor = 0
    for (start, stop), rank in sorted((tuple(r["shard"]), r["rank"]) for r in records):
        if start != cursor:
            failures.append(
                f"{arm} rank {rank}: shard coverage gap/overlap — shard starts at {start}, "
                f"expected {cursor}"
            )
        cursor = max(cursor, stop)
    total = cursor

    for r in records:
        rank = r["rank"]
        start, stop = r["shard"]
        if stop - start != r["galaxies"]:
            failures.append(
                f"{arm} rank {rank}: shard {list(r['shard'])} spans {stop - start} galaxies "
                f"but the record says {r['galaxies']}"
            )
        written = r.get("records_written")
        hits = r.get("cache_hits", 0)
        if written + hits != r["galaxies"]:
            failures.append(
                f"{arm} rank {rank}: wrote {written} records and reused {hits} contexts, "
                f"accounting for {written + hits} of {r['galaxies']} galaxies"
            )
        if r.get("records_failed", 0):
            failures.append(f"{arm} rank {rank}: {r['records_failed']} failed record(s)")
        if r.get("throttle_events", 0):
            failures.append(
                f"{arm} rank {rank}: {r['throttle_events']} Bedrock throttle event(s) — "
                f"a throttled rank is indistinguishable from a slow one, run invalid"
            )
        if r.get("error"):
            failures.append(f"{arm} rank {rank}: error {r['error']}")

    return world_size, total


def check_run(armada_records, langchain_records, isolated=None):
    """Gate one grouped run.

    Every arm present must pass on its own and agree with the others on both
    the population covered and the agent count: a run is comparable only if its
    arms ran the same work at the same N. `isolated` is optional so a two-arm
    run still gates, but when the no-sharing control is present it is held to
    the same agreement as the rest — an uncompared control is not a control.
    """
    failures = []
    arms = [("armada", armada_records), ("langchain", langchain_records)]
    if isolated is not None:
        arms.append(("isolated", isolated))

    checked = [(name, _check_arm(name, records, failures)) for name, records in arms]
    present = [(name, result) for name, result in checked if result is not None]
    if len(present) < 2:
        return GateResult(passed=not failures, failures=failures)

    reference_name, (reference_n, reference_total) = present[0]
    for name, (world_size, total) in present[1:]:
        if world_size != reference_n:
            failures.append(
                f"world_size differs between arms: {reference_name} ran {reference_n} ranks, "
                f"{name} ran {world_size} — not a comparable run"
            )
        if total != reference_total:
            failures.append(
                f"coverage differs between arms: {reference_name} covered {reference_total} "
                f"galaxies, {name} covered {total}"
            )
    return GateResult(passed=not failures, failures=failures)