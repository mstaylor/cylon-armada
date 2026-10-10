"""Requests carry their reference inputs, and their parameters can be read back exactly."""

import pytest

from cosmic_campaign.grid import CampaignSettings, Configuration, RunSlot, reference_input, run_slots
from cosmic_campaign.requests import execution_requests, parse_parameters, sweep_requests

SETTINGS = CampaignSettings("b", "b", "Anomaly Detection", "folder",
                            {"A": "/tmp/a.py", "B": "/tmp/b.py"}, "p/exp2")
CONFIG = Configuration("scaling", 100, 1, 512, 11)


def test_every_execution_request_carries_its_reference_input():
    slots = run_slots([CONFIG], 1, 3)
    requests = execution_requests(slots, "A", SETTINGS, seed=7)
    assert [r.references for r in requests] == [(reference_input(s, "A", SETTINGS),) for s in slots]
    assert len({r.request_id for r in requests}) == len(requests)


def test_paraphrases_are_seeded_and_vary():
    slots = run_slots([CONFIG, Configuration("scaling", 25, 12.6, 512, 517)], 1, 3)
    first = [r.text for r in execution_requests(slots, "A", SETTINGS, seed=7)]
    again = [r.text for r in execution_requests(slots, "A", SETTINGS, seed=7)]
    assert first == again
    assert len(set(first)) > 1


@pytest.mark.parametrize("slot", [RunSlot(CONFIG, "warmup", 0), RunSlot(CONFIG, "measured", 3),
                                  RunSlot(Configuration("scaling", 25, 12.6, 512, 517), "measured", 2),
                                  RunSlot(Configuration("batch", 100, 1, 64, 11), "measured", 4)])
def test_parameters_read_back_exactly_from_every_template(slot):
    for seed in range(5):
        [request] = execution_requests([slot], "A", SETTINGS, seed=seed)
        params = parse_parameters(request.text)
        c = slot.configuration
        assert (params["partition_mb"], params["data_gb"], params["batch_size"]) == (c.partition_mb, c.data_gb, c.batch_size)
        assert (params["phase"], params["run_index"]) == (slot.phase, slot.index)
        assert params["series"] == c.series


def test_the_batch_sweep_run_at_batch_512_is_worded_apart_from_the_scaling_run():
    scaling = RunSlot(CONFIG, "measured", 1)
    batch = RunSlot(Configuration("batch", 100, 1, 512, 11), "measured", 1)
    for seed in range(5):
        [a] = execution_requests([scaling], "A", SETTINGS, seed=seed)
        [b] = execution_requests([batch], "A", SETTINGS, seed=seed)
        assert a.text != b.text
        assert (parse_parameters(a.text)["series"], parse_parameters(b.text)["series"]) == ("scaling", "batch")


def test_a_request_without_a_partition_size_is_rejected_by_name():
    with pytest.raises(ValueError, match="partition"):
        parse_parameters("run inference on 1 GB with batch size 512, measured run 1")


def test_a_sweep_request_expands_to_every_slot_of_its_series():
    configs = [Configuration("scaling", 100, d, 512, w) for d, w in ((1, 11), (2, 21))]
    [request] = sweep_requests(configs, 1, 3, "A", SETTINGS, seed=1)
    assert request.kind == "sweep" and len(request.slots) == 8
    assert "100 MB" in request.text
