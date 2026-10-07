"""The Cosmic AI campaign: configurations, run slots and reference Step Functions inputs.

Reproduces AI-for-Astronomy/aws/results: partition sizes x data sizes at batch 512, a batch
sweep at 100 MB / 1 GB, plus 1 and 2 worker baselines. reference_input is the known-correct
input for every execution; Experiment 2's agents are checked against it.
"""

import math
from dataclasses import dataclass

SERIES = ("scaling", "batch", "baseline")
PHASES = ("warmup", "measured")


def num_workers(data_gb, partition_mb):
    return math.ceil(data_gb * 1024 / partition_mb)


@dataclass(frozen=True)
class Configuration:
    series: str
    partition_mb: int
    data_gb: float
    batch_size: int
    workers: int


@dataclass(frozen=True)
class RunSlot:
    configuration: Configuration
    phase: str
    index: int


@dataclass(frozen=True)
class CampaignSettings:
    bucket: str
    data_bucket: str
    object_name: str
    object_type: str
    scripts: dict
    result_prefix: str


def scaling_grid(partitions, data_sizes, batch_size):
    return [Configuration("scaling", p, d, batch_size, num_workers(d, p))
            for p in partitions for d in data_sizes]


def batch_sweep(partition_mb, data_gb, batch_sizes):
    workers = num_workers(data_gb, partition_mb)
    return [Configuration("batch", partition_mb, data_gb, b, workers) for b in batch_sizes]


def baseline_configurations(partition_mb, worker_counts, batch_size):
    return [Configuration("baseline", partition_mb, round(n * partition_mb / 1024, 4), batch_size, n)
            for n in worker_counts]


def run_slots(configurations, warmup_runs, measured_runs):
    slots = []
    for configuration in configurations:
        slots += [RunSlot(configuration, "warmup", k) for k in range(warmup_runs)]
        slots += [RunSlot(configuration, "measured", k) for k in range(1, measured_runs + 1)]
    return slots


def format_gb(data_gb):
    text = f"{round(data_gb, 4):.4f}".rstrip("0").rstrip(".")
    return f"{text}GB"


def _run_folder(slot):
    return f"warmup{slot.index}" if slot.phase == "warmup" else f"run{slot.index}"


def result_path(result_prefix, arm, slot):
    c = slot.configuration
    base = f"{result_prefix}/{arm}/result-partition-{c.partition_mb}MB"
    if c.series == "baseline":
        return f"{base}/baseline-ws{c.workers}/{_run_folder(slot)}"
    if c.series == "batch":
        return f"{base}/{format_gb(c.data_gb)}/Batches/batch{c.batch_size}/{_run_folder(slot)}"
    return f"{base}/{format_gb(c.data_gb)}/{_run_folder(slot)}"


def reference_input(slot, arm, settings):
    if arm not in settings.scripts:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(settings.scripts)}")
    c = slot.configuration
    return {
        "bucket": settings.bucket,
        "file_limit": str(c.workers),
        "world_size": c.workers,
        "batch_size": c.batch_size,
        "object_type": settings.object_type,
        "S3_object_name": settings.object_name,
        "script": settings.scripts[arm],
        "result_path": result_path(settings.result_prefix, arm, slot),
        "data_bucket": settings.data_bucket,
        "data_prefix": f"{c.partition_mb}MB",
    }
