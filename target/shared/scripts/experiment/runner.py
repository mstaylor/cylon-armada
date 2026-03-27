"""Experiment Runner — drives local and AWS-deployed benchmarks.

Runs a matrix of experiments varying:
  - Task count (scaling dimension)
  - Similarity threshold (reuse sensitivity)
  - Embedding dimensions (256/512/1024)
  - SIMD backend (NUMPY, CYTHON_BATCH, PYCYLON)
  - Baseline vs. context-reuse

Each experiment records cost, latency, reuse rate, and cache hit metrics.
Results are written to JSON files in the output directory for analysis.
"""

import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from itertools import product
from typing import Optional

# Add shared scripts to path so imports resolve when running standalone
# runner.py lives in target/shared/scripts/experiment/ — parent is target/shared/scripts/
_scripts_dir = os.path.join(os.path.dirname(__file__), '..')
if os.path.isdir(_scripts_dir) and os.path.abspath(_scripts_dir) not in sys.path:
    sys.path.insert(0, os.path.abspath(_scripts_dir))

# Add python bindings AFTER scripts so context.embedding (scripts) isn't shadowed
# by context.context_table (python bindings)
_python_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'python')
if os.path.isdir(_python_dir) and os.path.abspath(_python_dir) not in sys.path:
    sys.path.append(os.path.abspath(_python_dir))

from coordinator.agent_coordinator import AgentCoordinator
from context.router import SIMDBackend
from cost.bedrock_pricing import BedrockConfig
from experiment.benchmark import ExperimentBenchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default task set — used when no custom tasks are provided
# ---------------------------------------------------------------------------

DEFAULT_TASKS = [
    "Summarize the key benefits of serverless computing for enterprise applications",
    "List the advantages of using serverless architecture in large organizations",
    "Explain the cost model of AWS Lambda for production workloads",
    "Describe the pricing structure of serverless functions on AWS",
    "Compare serverless vs container-based deployments for microservices",
    "What are the trade-offs between Lambda and ECS for microservice architectures",
    "How does cold start latency affect serverless application performance",
    "Explain the impact of cold starts on AWS Lambda response times",
    "Describe best practices for monitoring serverless applications",
    "What observability tools work best with AWS Lambda deployments",
    "How do you handle state management in serverless workflows",
    "Explain patterns for managing state across Lambda function invocations",
    "What security considerations apply to serverless architectures",
    "Describe security best practices for AWS Lambda functions",
    "How does auto-scaling work differently in serverless vs traditional deployments",
    "Compare scaling behavior of Lambda with EC2 auto-scaling groups",
]


def load_tasks(tasks_file: str) -> list[str]:
    """Load tasks from a JSON file.

    Expected format — either a flat list of strings:
        ["task one", "task two", ...]

    Or an object with a "tasks" key:
        {"tasks": ["task one", "task two", ...]}
    """
    with open(tasks_file) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]

    raise ValueError(
        f"Invalid tasks file format. Expected a JSON list or object with 'tasks' key, "
        f"got {type(data).__name__}"
    )


def sample_tasks(
    tasks: list[str],
    count: int,
    strategy: str = "stratified",
    seed: int = 42,
) -> list[str]:
    """Select a subset of tasks using the specified sampling strategy.

    Args:
        tasks: Full task list.
        count: Number of tasks to select.
        strategy: Sampling strategy:
            - "stratified": Pick evenly across the task list, ensuring
              coverage of all categories (default).
            - "sequential": Take the first N tasks (for debugging).
            - "random": Uniform random sample.
        seed: Random seed for reproducibility (used by stratified and random).

    Returns:
        Selected task list of length min(count, len(tasks)).
    """
    if count >= len(tasks):
        return list(tasks)

    if strategy == "sequential":
        return tasks[:count]

    if strategy == "random":
        import random
        rng = random.Random(seed)
        return rng.sample(tasks, count)

    # Stratified: pick evenly spaced indices across the full list.
    # For 4 tasks from 32: picks indices 0, 8, 16, 24 — one from each quarter.
    # This ensures coverage across all categories in structured scenario files.
    import random
    rng = random.Random(seed)
    step = len(tasks) / count
    indices = [int(i * step) for i in range(count)]

    # Add jitter within each stratum so repeated runs with different seeds
    # don't always pick the exact same task from each category.
    stratum_size = max(1, int(step))
    sampled_indices = []
    for base_idx in indices:
        stratum_start = base_idx
        stratum_end = min(base_idx + stratum_size, len(tasks))
        sampled_indices.append(rng.randint(stratum_start, stratum_end - 1))

    # Deduplicate while preserving order (rare edge case with small lists)
    seen = set()
    unique = []
    for idx in sampled_indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)

    return [tasks[i] for i in unique]


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    task_count: int
    similarity_threshold: float
    embedding_dimensions: int
    backend: str = "NUMPY"
    context_backend: str = os.environ.get("CONTEXT_BACKEND", "cylon")
    baseline: bool = False
    sampling_strategy: str = "stratified"  # "stratified", "sequential", "random"
    seed: int = 42
    llm_model_id: str = "amazon.nova-lite-v1:0"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    region: str = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    redis_host: str = os.environ.get("REDIS_HOST", "localhost")
    redis_port: int = int(os.environ.get("REDIS_PORT", "6379"))
    dynamo_endpoint_url: Optional[str] = os.environ.get("DYNAMO_ENDPOINT_URL", "http://localhost:8000")


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    config: dict
    workflow_result: dict
    wall_clock_ms: float
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


def run_experiment(
    config: ExperimentConfig,
    tasks: Optional[list[str]] = None,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    output_dir: str = "target/shared/scripts/experiment/results",
) -> ExperimentResult:
    """Run a single experiment locally with StopWatch instrumentation.

    Args:
        config: Experiment configuration.
        tasks: Custom task list. If None, uses DEFAULT_TASKS.
        s3_bucket: S3 bucket for result upload. None = local only.
        s3_prefix: S3 key prefix.
        output_dir: Local output directory for benchmark files.
    """
    # StopWatch benchmark — clear previous timers
    bench = ExperimentBenchmark(
        name=config.name,
        s3_bucket=s3_bucket,
        s3_prefix=s3_prefix,
    )

    bedrock_config = BedrockConfig.resolve(payload={
        "llm_model_id": config.llm_model_id,
        "embedding_model_id": config.embedding_model_id,
        "embedding_dimensions": config.embedding_dimensions,
        "similarity_threshold": config.similarity_threshold,
        "region": config.region,
        "context_backend": config.context_backend,
    })

    coordinator = AgentCoordinator(config=bedrock_config)
    all_tasks = tasks or DEFAULT_TASKS
    task_list = sample_tasks(
        all_tasks, config.task_count,
        strategy=config.sampling_strategy,
        seed=config.seed,
    )
    backend = SIMDBackend[config.backend]

    logger.info(
        "Running experiment: %s | tasks=%d threshold=%.2f dims=%d backend=%s baseline=%s",
        config.name, len(task_list), config.similarity_threshold,
        config.embedding_dimensions, config.backend, config.baseline,
    )

    bench.start("total")
    result = coordinator.run_local(
        tasks=task_list,
        backend=backend,
        dynamo_endpoint_url=config.dynamo_endpoint_url,
        redis_host=config.redis_host,
        redis_port=config.redis_port,
        baseline=config.baseline,
    )
    bench.stop("total")

    wall_clock_ms = bench.elapsed_ms("total")

    # Record experiment metrics for the results pipeline
    cost_summary = result.get("cost_summary", {})
    reuse_stats = result.get("reuse_stats", {})
    bench.record("total_cost", cost_summary.get("total_cost", 0))
    bench.record("baseline_cost", cost_summary.get("baseline_cost", 0))
    bench.record("savings_pct", cost_summary.get("savings_pct", 0))
    bench.record("reuse_rate", reuse_stats.get("reuse_rate", 0))
    bench.record("cache_hits", reuse_stats.get("cache_hits", 0))
    bench.record("llm_calls", reuse_stats.get("llm_calls", 0))
    bench.record("task_count", len(task_list))
    bench.record("similarity_threshold", config.similarity_threshold)
    bench.record("embedding_dimensions", config.embedding_dimensions)
    bench.record("backend", config.backend)
    bench.record("baseline", config.baseline)

    # Save benchmark files (locally + optional S3)
    bench.save(output_dir)

    return ExperimentResult(
        config=asdict(config),
        workflow_result=result,
        wall_clock_ms=round(wall_clock_ms, 2),
    )


def run_experiment_matrix(
    task_counts: list[int] = None,
    thresholds: list[float] = None,
    dimensions: list[int] = None,
    backends: list[str] = None,
    context_backend: Optional[str] = None,
    output_dir: str = "target/shared/scripts/experiment/results",
    include_baseline: bool = True,
    tasks: Optional[list[str]] = None,
    sampling_strategy: str = "stratified",
    seed: int = 42,
    s3_bucket: Optional[str] = None,
    s3_prefix: Optional[str] = None,
) -> list[ExperimentResult]:
    """Run a matrix of experiments and save results.

    Args:
        task_counts: List of task counts to test (default: [4, 8, 16])
        thresholds: List of similarity thresholds (default: [0.7, 0.8, 0.9])
        dimensions: List of embedding dimensions (default: [256, 1024])
        backends: List of SIMD backends (default: ["NUMPY"])
        output_dir: Directory for result JSON files
        include_baseline: Whether to include a baseline (no reuse) run per config
        tasks: Custom task list. If None, uses DEFAULT_TASKS.
        sampling_strategy: How to select tasks: "stratified" (default), "sequential", "random".
        seed: Random seed for reproducible sampling (default: 42).

    Returns:
        List of ExperimentResult objects
    """
    if task_counts is None:
        task_counts = [4, 8, 16]
    if thresholds is None:
        thresholds = [0.7, 0.8, 0.9]
    if dimensions is None:
        dimensions = [256, 1024]
    if backends is None:
        backends = ["NUMPY"]

    os.makedirs(output_dir, exist_ok=True)
    results = []
    run_id = str(uuid.uuid4())[:8]

    # context_backend kwargs — only set if explicitly provided (otherwise ExperimentConfig
    # reads from CONTEXT_BACKEND env var)
    cb_kwargs = {"context_backend": context_backend} if context_backend else {}

    configs = []
    for tc, thresh, dim, be in product(task_counts, thresholds, dimensions, backends):
        cb = cb_kwargs.get("context_backend") or os.environ.get("CONTEXT_BACKEND", "cylon")
        # Context-reuse run
        configs.append(ExperimentConfig(
            name=f"reuse_t{tc}_th{thresh}_d{dim}_{cb}_{be}",
            task_count=tc,
            similarity_threshold=thresh,
            embedding_dimensions=dim,
            backend=be,
            baseline=False,
            sampling_strategy=sampling_strategy,
            seed=seed,
            **cb_kwargs,
        ))

        # Baseline run (no reuse) — run immediately after reuse so baseline cost
        # is available for savings comparison in the summary
        if include_baseline:
            configs.append(ExperimentConfig(
                name=f"baseline_t{tc}_d{dim}_{cb}_{be}",
                task_count=tc,
                similarity_threshold=1.0,
                embedding_dimensions=dim,
                backend=be,
                baseline=True,
                sampling_strategy=sampling_strategy,
                seed=seed,
                **cb_kwargs,
            ))

    logger.info("Running %d experiments (run_id=%s)", len(configs), run_id)

    for i, config in enumerate(configs):
        logger.info("Experiment %d/%d: %s", i + 1, len(configs), config.name)
        try:
            ExperimentBenchmark.clear()
            result = run_experiment(
                config, tasks=tasks,
                s3_bucket=s3_bucket, s3_prefix=s3_prefix,
                output_dir=output_dir,
            )
            results.append(result)

            # Write individual result
            result_path = os.path.join(output_dir, f"{run_id}_{config.name}.json")
            with open(result_path, "w") as f:
                json.dump(asdict(result), f, indent=2, default=str)

        except Exception as e:
            logger.error("Experiment %s failed: %s", config.name, e)
            results.append(ExperimentResult(
                config=asdict(config),
                workflow_result={"error": str(e)},
                wall_clock_ms=0,
            ))

    # Write summary
    summary_path = os.path.join(output_dir, f"{run_id}_summary.json")
    summary = {
        "run_id": run_id,
        "total_experiments": len(results),
        "successful": sum(1 for r in results if "error" not in r.workflow_result),
        "results": [asdict(r) for r in results],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("All experiments complete. Summary: %s", summary_path)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run context-reuse experiments")
    parser.add_argument("--tasks", type=int, nargs="+", default=[4, 8],
                        help="Task counts to test")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.8],
                        help="Similarity thresholds")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[256],
                        help="Embedding dimensions")
    parser.add_argument("--backends", type=str, nargs="+", default=["NUMPY"],
                        help="SIMD backends")
    parser.add_argument("--context-backend", type=str, default=None,
                        choices=["cylon", "redis"],
                        help="Context store backend (default: from CONTEXT_BACKEND env or 'cylon')")
    parser.add_argument("--output", type=str, default="target/shared/scripts/experiment/results",
                        help="Output directory")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline runs")
    parser.add_argument("--tasks-file", type=str, default=None,
                        help="Path to JSON file with custom tasks (list of strings)")
    parser.add_argument("--sampling", type=str, default="stratified",
                        choices=["stratified", "sequential", "random"],
                        help="Task sampling strategy (default: stratified)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--s3-bucket", type=str, default=None,
                        help="S3 bucket for result upload (omit for local-only)")
    parser.add_argument("--s3-prefix", type=str, default="experiments",
                        help="S3 key prefix for results (default: experiments)")

    # cosmic-ai: generate tasks from real astronomical inference
    cosmic = parser.add_argument_group("cosmic-ai", "Generate tasks from SDSS inference")
    cosmic.add_argument("--cosmic-ai", action="store_true",
                        help="Run with tasks generated from AstroMAE inference")
    cosmic.add_argument("--data-path", type=str, default=None,
                        help="Path to SDSS .pt data partition")
    cosmic.add_argument("--model-path", type=str, default=None,
                        help="Path to pre-trained AstroMAE model checkpoint")
    cosmic.add_argument("--inference-batch-size", type=int, default=512,
                        help="Batch size for AstroMAE inference")
    cosmic.add_argument("--inference-device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device for inference")

    args = parser.parse_args()

    custom_tasks = None

    if args.cosmic_ai:
        if not args.data_path or not args.model_path:
            parser.error("--cosmic-ai requires --data-path and --model-path")

        from cosmic_ai.task_generator import generate_tasks_from_data

        custom_tasks, inference_results = generate_tasks_from_data(
            data_path=args.data_path,
            model_path=args.model_path,
            batch_size=args.inference_batch_size,
            device=args.inference_device,
            max_tasks=max(args.tasks) if args.tasks else 16,
            seed=42,
        )
        logger.info(
            "cosmic-ai: generated %d tasks from %d inference results",
            len(custom_tasks), inference_results["metrics"]["num_samples"],
        )

    elif args.tasks_file:
        custom_tasks = load_tasks(args.tasks_file)
        logger.info("Loaded %d tasks from %s", len(custom_tasks), args.tasks_file)

    run_experiment_matrix(
        task_counts=args.tasks,
        thresholds=args.thresholds,
        dimensions=args.dimensions,
        backends=args.backends,
        context_backend=args.context_backend,
        output_dir=args.output,
        include_baseline=not args.no_baseline,
        tasks=custom_tasks,
        sampling_strategy=args.sampling,
        seed=args.seed,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
    )