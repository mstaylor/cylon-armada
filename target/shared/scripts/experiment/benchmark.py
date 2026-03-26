"""Experiment benchmarking — StopWatch + optional S3 upload.

Uses cloudmesh.common.StopWatch for structured timing.
Produces CSV output compatible with Cylon's results pipeline.

S3 upload is controlled by configuration:
  - S3_RESULTS_BUCKET env var or --s3-bucket CLI arg: enables upload
  - If not set, results are only written locally

Usage:
    from experiment.benchmark import ExperimentBenchmark

    bench = ExperimentBenchmark("reuse_t4_th0.8_d256", s3_bucket="my-bucket")
    bench.start("embedding")
    # ... do embedding ...
    bench.stop("embedding")
    bench.record("reuse_rate", 65.0)
    bench.save(output_dir="results/")
"""

import json
import logging
import os
from typing import Optional

import pandas as pd
from cloudmesh.common.StopWatch import StopWatch

logger = logging.getLogger(__name__)


class ExperimentBenchmark:
    """Structured timing and metrics for context-reuse experiments.

    Args:
        name: Experiment name (used as StopWatch tag and file prefix).
        s3_bucket: S3 bucket for result upload. None = local only.
        s3_prefix: S3 key prefix (e.g., "experiments/run_id").
    """

    def __init__(
        self,
        name: str,
        s3_bucket: Optional[str] = None,
        s3_prefix: Optional[str] = None,
    ):
        self.name = name
        self.s3_bucket = s3_bucket or os.environ.get("S3_RESULTS_BUCKET")
        self.s3_prefix = s3_prefix or os.environ.get("S3_RESULTS_PREFIX", "experiments")
        self._metrics = {}

    def start(self, phase: str):
        """Start a named timer phase."""
        StopWatch.start(f"{self.name}_{phase}")

    def stop(self, phase: str) -> float:
        """Stop a named timer phase. Returns elapsed seconds."""
        StopWatch.stop(f"{self.name}_{phase}")
        return StopWatch.get(f"{self.name}_{phase}")

    def elapsed(self, phase: str) -> float:
        """Get elapsed seconds for a phase."""
        return StopWatch.get(f"{self.name}_{phase}")

    def elapsed_ms(self, phase: str) -> float:
        """Get elapsed milliseconds for a phase."""
        return StopWatch.get(f"{self.name}_{phase}") * 1000

    def record(self, key: str, value):
        """Record a named metric (cost, reuse rate, etc.)."""
        self._metrics[key] = value

    def get_all_timings(self) -> dict:
        """Get all timer results as {phase: elapsed_s}."""
        timings = {}
        for key in StopWatch.keys():
            if key.startswith(f"{self.name}_"):
                phase = key[len(self.name) + 1:]
                timings[phase] = StopWatch.get(key)
        return timings

    def to_dict(self) -> dict:
        """Export all timings and metrics as a flat dict."""
        result = {"experiment_name": self.name}
        for phase, elapsed_s in self.get_all_timings().items():
            result[f"{phase}_s"] = round(elapsed_s, 6)
            result[f"{phase}_ms"] = round(elapsed_s * 1000, 4)
        result.update(self._metrics)
        return result

    def save(self, output_dir: str) -> dict:
        """Save results locally. Optionally upload to S3.

        Produces three files:
        - {name}_stopwatch.csv  — cloudmesh benchmark (system info + timings)
        - {name}_summary.csv    — data-only summary (like scaling.py output)
        - {name}_metrics.json   — full metrics + timings as JSON

        Args:
            output_dir: Local directory for output files.

        Returns:
            dict with 'stopwatch_csv', 'summary_csv', 'json',
            and optionally 's3_keys' paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # StopWatch CSV (cloudmesh format — includes machine info)
        stopwatch_path = os.path.join(output_dir, f"{self.name}_stopwatch.csv")
        StopWatch.benchmark(tag=self.name, filename=stopwatch_path)
        paths["stopwatch_csv"] = stopwatch_path

        # Summary CSV (data only — matches scaling.py pattern)
        summary_path = os.path.join(output_dir, f"{self.name}_summary.csv")
        self._write_summary_csv(summary_path)
        paths["summary_csv"] = summary_path

        # Metrics JSON
        json_path = os.path.join(output_dir, f"{self.name}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        paths["json"] = json_path

        logger.info("Benchmark saved: %s, %s, %s",
                     stopwatch_path, summary_path, json_path)

        # Optional S3 upload
        if self.s3_bucket:
            paths["s3_keys"] = self._upload_to_s3(
                stopwatch_path, summary_path, json_path)

        return paths

    def _write_summary_csv(self, path: str):
        """Write a data-only summary CSV (no machine info).

        Columns: experiment_name, phase timings (ms), and all recorded metrics.
        Compatible with Cylon's results pipeline aggregator.
        """
        row = {"experiment_name": self.name}

        # Timing columns (milliseconds)
        for phase, elapsed_s in self.get_all_timings().items():
            row[f"{phase}_ms"] = round(elapsed_s * 1000, 4)

        # Metric columns
        row.update(self._metrics)

        df = pd.DataFrame([row])
        df.to_csv(path, index=False)
        logger.debug("Summary CSV: %s", path)

    def _upload_to_s3(self, *file_paths) -> list[str]:
        """Upload files to S3."""
        import boto3
        from botocore.exceptions import ClientError

        s3_client = boto3.client("s3")
        uploaded = []

        for local_path in file_paths:
            filename = os.path.basename(local_path)
            s3_key = f"{self.s3_prefix}/{self.name}/{filename}"
            try:
                s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                uploaded.append(s3_key)
                logger.info("Uploaded s3://%s/%s", self.s3_bucket, s3_key)
            except ClientError as e:
                logger.warning("S3 upload failed for %s: %s", filename, e)

        return uploaded

    @staticmethod
    def clear():
        """Clear all StopWatch timers (between experiments)."""
        StopWatch.clear()