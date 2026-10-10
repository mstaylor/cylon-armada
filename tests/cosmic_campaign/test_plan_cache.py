import numpy as np

from cosmic_campaign.grid import CampaignSettings, Configuration, RunSlot
from cosmic_campaign.plan_cache import CacheEntry, CacheMode, PlanCache
from cosmic_campaign.plan_check import mismatched_fields
from cosmic_campaign.requests import execution_requests

SETTINGS = CampaignSettings("b", "b", "Anomaly Detection", "folder", {"A": "/tmp/a.py"}, "p/exp2")
ONE_GB, TWO_GB = Configuration("scaling", 100, 1, 512, 11), Configuration("scaling", 100, 2, 512, 21)
BATCH_512 = Configuration("batch", 100, 1, 512, 11)


def _req(config, phase="measured", index=1):
    return execution_requests([RunSlot(config, phase, index)], "A", SETTINGS, seed=0)[0]


def _cache(mode, plans=None):
    cache = PlanCache(mode, similarity_threshold=0.9, settings=SETTINGS, arm="A")
    first = _req(ONE_GB, index=1)
    cache.add(CacheEntry(first.text, np.array([1.0, 0.0]),
                         [first.references[0]] if plans is None else plans, origin_rank=0))
    return cache


def test_off_never_hits_and_reports_no_match():
    assert _cache(CacheMode.OFF).lookup(_req(ONE_GB, index=2).text, np.array([1.0, 0.0])) == (None, {})


def test_exact_reuse_across_runs_serves_the_wrong_result_path():
    later = _req(ONE_GB, index=2)
    outcome, _ = _cache(CacheMode.EXACT).lookup(later.text, np.array([1.0, 0.0]))
    assert outcome.source == "cache_exact"
    assert outcome.plans[0]["result_path"] != later.references[0]["result_path"]


def test_exact_reuse_between_neighbouring_sizes_serves_the_wrong_worker_count():
    neighbour = _req(TWO_GB, index=1)
    outcome, _ = _cache(CacheMode.EXACT).lookup(neighbour.text, np.array([0.99, 0.14]))
    assert outcome.plans[0]["file_limit"] != neighbour.references[0]["file_limit"]


def test_structural_reuse_of_a_correct_plan_fills_every_varying_field_from_the_request():
    for target in (_req(ONE_GB, index=2), _req(TWO_GB, "warmup", 0), _req(BATCH_512, index=3)):
        outcome, _ = _cache(CacheMode.STRUCTURAL).lookup(target.text, np.array([0.99, 0.14]))
        assert outcome.source == "cache_structural"
        assert outcome.plans == [target.references[0]]


def test_structural_reuse_carries_a_wrong_cached_plan_into_the_new_plan():
    first = _req(ONE_GB, index=1).references[0]
    wrong = {**first, "bucket": "wrong-bucket", "script": "/tmp/other.py",
             "result_path": first["result_path"].replace("p/exp2/A", "q/A")}
    target = _req(TWO_GB, index=2)
    outcome, _ = _cache(CacheMode.STRUCTURAL, plans=[wrong]).lookup(target.text, np.array([0.99, 0.14]))
    assert set(mismatched_fields(outcome.plans[0], target.references[0])) == {"bucket", "script", "result_path"}


def test_structural_reuse_of_an_empty_cached_plan_is_a_miss():
    outcome, match = _cache(CacheMode.STRUCTURAL, plans=[]).lookup(_req(TWO_GB).text, np.array([1.0, 0.0]))
    assert outcome is None and match["similarity"] == 1.0


def test_lookup_reports_the_similarity_and_the_matched_entry_for_hits_and_misses():
    cache = _cache(CacheMode.EXACT)
    _, hit = cache.lookup(_req(ONE_GB, index=2).text, np.array([1.0, 0.0]))
    assert hit == {"similarity": 1.0, "matched_request": _req(ONE_GB, index=1).text, "matched_origin_rank": 0}
    outcome, miss = cache.lookup(_req(TWO_GB).text, np.array([0.0, 1.0]))
    assert outcome is None and miss["similarity"] == 0.0


def test_dissimilar_requests_miss():
    outcome, _ = _cache(CacheMode.EXACT).lookup(_req(TWO_GB).text, np.array([0.0, 1.0]))
    assert outcome is None


def test_entries_round_trip_through_arrow_for_allgather():
    cache = _cache(CacheMode.EXACT)
    table = cache.to_arrow(cache.entries)
    [entry] = PlanCache.entries_from_arrow(table)
    assert entry.plans == cache.entries[0].plans and entry.origin_rank == 0
    assert np.allclose(entry.embedding, [1.0, 0.0])


def test_the_shared_table_uses_only_flat_column_types():
    """pycylon's AllGather silently returns nothing for list-typed columns."""
    import pyarrow as pa

    cache = _cache(CacheMode.EXACT)
    table = PlanCache.to_arrow(cache.entries)
    assert not any(pa.types.is_list(f.type) or pa.types.is_large_list(f.type) for f in table.schema)


def test_no_new_entries_still_sends_one_row_that_receivers_skip():
    """An empty table crashes pycylon's AllGather, so a round with nothing new sends a marker row."""
    table = PlanCache.to_arrow([])
    assert table.num_rows == 1
    assert PlanCache.entries_from_arrow(table) == []