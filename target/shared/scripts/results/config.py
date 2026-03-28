"""
Configuration for the cylon-armada experiment results pipeline.

Follows the cylon results pipeline pattern — ExperimentConfig defines
one experiment (platform + scaling + instance), PipelineConfig defines
the full pipeline.

Metric columns are specific to context-reuse experiments:
cost savings, reuse rate, search/LLM latency, etc.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions — cylon-armada metrics
# ---------------------------------------------------------------------------

# Timing columns (milliseconds in raw data, converted to seconds in output)
TIMING_COLUMNS = [
    "total_ms",
    "search_latency_ms",
    "llm_latency_ms",
]

# Cost columns (USD, no unit conversion)
COST_COLUMNS = [
    "total_cost",
    "baseline_cost",
    "savings_pct",
]

# Reuse / hit-rate columns (counts and percentages, no unit conversion)
REUSE_COLUMNS = [
    "reuse_rate",
    "cache_hits",
    "llm_calls",
]

# All metric columns to aggregate (compute mean/std across runs)
METRIC_COLUMNS = TIMING_COLUMNS + COST_COLUMNS + REUSE_COLUMNS

# Columns that are in milliseconds and should be converted to seconds
MS_TO_S_COLUMNS = [
    "total_ms",
    "search_latency_ms",
    "llm_latency_ms",
]

# Experiment parameter columns (used for grouping, not aggregated)
PARAM_COLUMNS = [
    "experiment_name",
    "task_count",
    "similarity_threshold",
    "embedding_dimensions",
    "backend",         # SIMD backend: NUMPY, CYTHON_BATCH, PYCYLON
    "context_backend", # Context store: cylon, redis
    "baseline",
]


@dataclass
class ExperimentConfig:
    """Defines one experiment: platform + configuration + data source."""
    platform: str               # "lambda", "fargate", "ecs", "rivanna"
    instance_label: str         # "1024MB", "g4dn.xlarge", "a100", etc.
    instance_detail: str        # Human-readable: "Lambda 1024MB", "g4dn.xlarge 4vCPU 16GB T4"
    node_counts: List[int]      # [1, 2, 4, 8, 16] — number of concurrent workers
    task_counts: List[int]      # [4, 8, 16, 32] — tasks per experiment
    thresholds: List[float] = field(default_factory=lambda: [0.7, 0.8, 0.9])
    dimensions: List[int] = field(default_factory=lambda: [256, 512, 1024])
    simd_backends: List[str] = field(default_factory=lambda: ["NUMPY"])
    context_backends: List[str] = field(default_factory=lambda: ["redis"])
    runs_per_config: int = 3    # Number of runs for std dev
    color: str = "blue"
    marker: str = "o"

    # Data source — one of these must be set
    s3_prefix_pattern: Optional[str] = None
    local_data_dir: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.platform.upper()} - {self.instance_detail}"

    @property
    def sheet_name(self) -> str:
        return f"{self.platform.upper()}"


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    s3_bucket: str = ""
    download_dir: str = "./data/raw"
    output_dir: str = "./output"
    chart_format: str = "svg"
    chart_dpi: int = 300
    experiments: List[ExperimentConfig] = field(default_factory=list)
    notebook_name: str = "context_reuse_results"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        import yaml
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        experiments = []
        for exp_data in data.get("experiments", []):
            experiments.append(ExperimentConfig(**exp_data))

        return cls(
            s3_bucket=data.get("s3_bucket", ""),
            download_dir=data.get("download_dir", "./data/raw"),
            output_dir=data.get("output_dir", "./output"),
            chart_format=data.get("chart_format", "svg"),
            chart_dpi=data.get("chart_dpi", 300),
            experiments=experiments,
            notebook_name=data.get("notebook_name", "context_reuse_results"),
        )

    @classmethod
    def from_args(cls, args) -> "PipelineConfig":
        """Build config from CLI arguments for single-experiment mode."""
        exp = ExperimentConfig(
            platform=args.platform,
            instance_label=args.instance,
            instance_detail=args.instance,
            node_counts=[int(n) for n in args.nodes.split(",")],
            task_counts=[int(t) for t in args.task_counts.split(",")],
            s3_prefix_pattern=getattr(args, "s3_prefix", None),
            local_data_dir=getattr(args, "local_dir", None),
        )
        return cls(
            s3_bucket=getattr(args, "bucket", "") or "",
            download_dir=getattr(args, "download_dir", "./data/raw"),
            output_dir=getattr(args, "output_dir", "./output"),
            experiments=[exp],
        )