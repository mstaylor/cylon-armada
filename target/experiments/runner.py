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
import time
import uuid
from dataclasses import dataclass, field, asdict
from itertools import product
from typing import Optional

from coordinator.agent_coordinator import AgentCoordinator
from context.router import SIMDBackend
from cost.bedrock_pricing import BedrockConfig

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


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    task_count: int
    similarity_threshold: float
    embedding_dimensions: int
    backend: str = "NUMPY"
    baseline: bool = False
    llm_model_id: str = "amazon.nova-lite-v1:0"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    region: str = "us-east-1"
    redis_host: str = "localhost"
    redis_port: int = 6379
    dynamo_endpoint_url: Optional[str] = "http://localhost:8000"


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
) -> ExperimentResult:
    """Run a single experiment locally.

    Args:
        config: Experiment configuration.
        tasks: Custom task list. If None, uses DEFAULT_TASKS.
    """
    bedrock_config = BedrockConfig(
        llm_model_id=config.llm_model_id,
        embedding_model_id=config.embedding_model_id,
        embedding_dimensions=config.embedding_dimensions,
        similarity_threshold=config.similarity_threshold,
        region=config.region,
    )

    coordinator = AgentCoordinator(config=bedrock_config)
    task_list = (tasks or DEFAULT_TASKS)[:config.task_count]
    backend = SIMDBackend[config.backend]

    logger.info(
        "Running experiment: %s | tasks=%d threshold=%.2f dims=%d backend=%s baseline=%s",
        config.name, len(task_list), config.similarity_threshold,
        config.embedding_dimensions, config.backend, config.baseline,
    )

    start = time.perf_counter()
    result = coordinator.run_local(
        tasks=task_list,
        backend=backend,
        dynamo_endpoint_url=config.dynamo_endpoint_url,
        redis_host=config.redis_host,
        redis_port=config.redis_port,
        baseline=config.baseline,
    )
    wall_clock_ms = (time.perf_counter() - start) * 1000

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
    output_dir: str = "target/experiments/results",
    include_baseline: bool = True,
    tasks: Optional[list[str]] = None,
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

    configs = []
    for tc, thresh, dim, be in product(task_counts, thresholds, dimensions, backends):
        # Context-reuse run
        configs.append(ExperimentConfig(
            name=f"reuse_t{tc}_th{thresh}_d{dim}_{be}",
            task_count=tc,
            similarity_threshold=thresh,
            embedding_dimensions=dim,
            backend=be,
            baseline=False,
        ))

        # Baseline run (no reuse)
        if include_baseline:
            configs.append(ExperimentConfig(
                name=f"baseline_t{tc}_d{dim}_{be}",
                task_count=tc,
                similarity_threshold=1.0,
                embedding_dimensions=dim,
                backend=be,
                baseline=True,
            ))

    logger.info("Running %d experiments (run_id=%s)", len(configs), run_id)

    for i, config in enumerate(configs):
        logger.info("Experiment %d/%d: %s", i + 1, len(configs), config.name)
        try:
            result = run_experiment(config, tasks=tasks)
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
    parser.add_argument("--output", type=str, default="target/experiments/results",
                        help="Output directory")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline runs")
    parser.add_argument("--tasks-file", type=str, default=None,
                        help="Path to JSON file with custom tasks (list of strings)")

    args = parser.parse_args()

    custom_tasks = None
    if args.tasks_file:
        custom_tasks = load_tasks(args.tasks_file)
        logger.info("Loaded %d tasks from %s", len(custom_tasks), args.tasks_file)

    run_experiment_matrix(
        task_counts=args.tasks,
        thresholds=args.thresholds,
        dimensions=args.dimensions,
        backends=args.backends,
        output_dir=args.output,
        include_baseline=not args.no_baseline,
        tasks=custom_tasks,
    )