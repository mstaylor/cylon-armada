"""Generate SLURM scripts for cylon-armada experiments on Rivanna.

Follows the cylon experiment-setup-apptainer.py pattern:
  1. Generates SLURM scripts from parameter combinations
  2. Each script runs inside the Apptainer container via runArmada.sh
  3. Uses --env to pass configuration, --bind for output and scripts

Env vars passed to the container align with armada_ecs_runner.py:
  TASKS_JSON, SCALING, WORLD_SIZE, RESULTS_DIR, EXPERIMENT_NAME,
  RESULTS_BUCKET, SIMD_BACKEND, CONTEXT_BACKEND, REDIS_HOST, REDIS_PORT,
  BEDROCK_LLM_MODEL_ID, BEDROCK_EMBEDDING_MODEL_ID,
  BEDROCK_EMBEDDING_DIMENSIONS, SIMILARITY_THRESHOLD,
  DYNAMO_TABLE_NAME, AWS_DEFAULT_REGION, CYLON_SESSION_ID,
  COMPUTE_PLATFORM, WORKFLOW_ID

Usage:
    # CPU smoke test (3 tasks, standard partition)
    python armada-experiment-setup.py --smoke \
        -d /scratch/$USER/cylon-armada/cylon-armada.sif \
        -r1 dev-cylon-redis1.aws-cylondata.com \
        -l1 /scratch/$USER/armada-results -l2 /output \
        -p2 standard

    # GPU smoke test (3 tasks, bii-gpu partition)
    python armada-experiment-setup.py --smoke --gpu \
        -d /scratch/$USER/cylon-armada/cylon-armada-gpu.sif \
        -r1 dev-cylon-redis1.aws-cylondata.com \
        -l1 /scratch/$USER/armada-results-gpu -l2 /output \
        -p2 bii-gpu

    # CPU full sweep (hydrology scenario)
    python armada-experiment-setup.py \
        -d /scratch/$USER/cylon-armada/cylon-armada.sif \
        -r1 dev-cylon-redis1.aws-cylondata.com \
        -n 1 -t 1 -c 10 \
        --tasks 4 8 16 32 \
        --thresholds 0.70 0.80 0.90 \
        --dimensions 256 512 1024 \
        --backends NUMPY PYCYLON \
        --runs 3 --scenario hydrology \
        -l1 /scratch/$USER/armada-results -l2 /output \
        -p2 standard

    # GPU full sweep (gcylon backend)
    python armada-experiment-setup.py --gpu \
        -d /scratch/$USER/cylon-armada/cylon-armada-gpu.sif \
        -r1 dev-cylon-redis1.aws-cylondata.com \
        -n 1 -t 1 -c 10 \
        --tasks 4 8 16 32 \
        --thresholds 0.70 0.80 0.90 \
        --dimensions 256 512 1024 \
        --backends GCYLON \
        --runs 3 --scenario hydrology --gpu \
        -l1 /scratch/$USER/armada-results-gpu -l2 /output \
        -p2 bii-gpu
"""

import json
import os
import uuid
import argparse
from textwrap import dedent

try:
    from cloudmesh.common.util import writefile
    from cloudmesh.common.util import banner
    from cloudmesh.common.console import Console
    HAS_CLOUDMESH = True
except ImportError:
    HAS_CLOUDMESH = False

    def writefile(filename, content):
        with open(filename, "w") as f:
            f.write(content)

    def banner(msg):
        print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")

    class Console:
        @staticmethod
        def ok(msg): print(f"[OK] {msg}")
        @staticmethod
        def error(msg): print(f"[ERROR] {msg}", flush=True)


# Smoke test task set — mirrors ECS smoke tests for direct comparison
SMOKE_TASKS = [
    "Summarize the benefits of context reuse in LLM pipelines",
    "Explain weak scaling in distributed systems",
    "Summarize the benefits of context reuse in LLM pipelines",
]

debug = False


def build_parser():
    parser = argparse.ArgumentParser(description="cylon-armada Rivanna experiment setup")

    # Smoke test shortcut
    parser.add_argument("--smoke", action="store_true",
                        help="Generate a single 3-task smoke test script")

    # SLURM configuration
    parser.add_argument("-n", dest="nodes", type=int, default=1)
    parser.add_argument("-t", dest="threads", type=int, default=1)
    parser.add_argument("-c", dest="cpus", type=int, default=10)
    parser.add_argument("-p2", dest="partition", type=str, default="standard")
    parser.add_argument("-m", dest="memory", type=str, default="DefMemPerNode")
    parser.add_argument("--time", dest="time_limit", type=str, default="4:00:00")
    parser.add_argument("--gpu", action="store_true", help="Request GPU resources")
    parser.add_argument("--gres", type=str, default="gpu:1")
    parser.add_argument("--account", type=str, default="bii_dsc_community")

    # Container
    parser.add_argument("-d", dest="docker_image", type=str, required=True,
                        help="Apptainer .sif image path")

    # Redis
    parser.add_argument("-r1", dest="redis_host", type=str,
                        default="dev-cylon-redis1.aws-cylondata.com")
    parser.add_argument("-p1", dest="redis_port", type=int, default=6379)

    # Experiment parameters (full sweep)
    parser.add_argument("--tasks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.8])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[256])
    parser.add_argument("--backends", type=str, nargs="+", default=["NUMPY"])
    parser.add_argument("--context-backend", type=str, default="cylon",
                        choices=["redis", "cylon"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--scenario", type=str, default="hydrology")
    parser.add_argument("--scaling", type=str, default="weak",
                        choices=["weak", "strong"])
    parser.add_argument("--world-size", type=int, default=1)

    # AWS / Bedrock
    parser.add_argument("--results-bucket", type=str, default="staylor.dev2")
    parser.add_argument("--dynamo-table", type=str, default="cylon-armada-context-store")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--llm-model", type=str, default="amazon.nova-lite-v1:0")
    parser.add_argument("--embedding-model", type=str,
                        default="amazon.titan-embed-text-v2:0")
    parser.add_argument("--session-id", type=str, default="cylon-armada")

    # Bind mounts
    parser.add_argument("-l1", dest="log_bind_host", type=str, required=True)
    parser.add_argument("-l2", dest="log_bind_container", type=str, default="/output")

    # Control

    parser.add_argument("--dry-run", action="store_true")

    return parser


def make_env_vars(args, tasks_json, exp_name, results_dir, backend, dim, threshold, world_size):
    """Build the env var list passed to the Apptainer container."""
    return [
        # Runner identity
        f"COMPUTE_PLATFORM={'rivanna-gpu' if args.gpu else 'rivanna'}",
        f"WORKFLOW_ID=rivanna-{exp_name}-{str(uuid.uuid4())[:8]}",
        f"CYLON_SESSION_ID={args.session_id}",
        # Tasks
        f"TASKS_JSON={tasks_json}",
        f"SCALING={args.scaling}",
        f"WORLD_SIZE={world_size}",
        # Results
        f"RESULTS_BUCKET={args.results_bucket}",
        f"RESULTS_DIR={results_dir}",
        f"EXPERIMENT_NAME={exp_name}",
        # SIMD / context
        f"SIMD_BACKEND={backend}",
        f"CONTEXT_BACKEND={args.context_backend}",
        # Redis
        f"REDIS_HOST={args.redis_host}",
        f"REDIS_PORT={args.redis_port}",
        # Bedrock
        f"BEDROCK_LLM_MODEL_ID={args.llm_model}",
        f"BEDROCK_EMBEDDING_MODEL_ID={args.embedding_model}",
        f"BEDROCK_EMBEDDING_DIMENSIONS={dim}",
        f"SIMILARITY_THRESHOLD={threshold}",
        f"DYNAMO_TABLE_NAME={args.dynamo_table}",
        f"AWS_DEFAULT_REGION={args.region}",
    ]


def make_slurm_script(args, exp_name, env_vars):
    """Render a SLURM script using the Apptainer pattern.

    Uses --env-file instead of --env so that TASKS_JSON (which contains
    commas) is not misinterpreted as an env var separator.
    Binds ~/.aws explicitly because --containall blocks home dir access.
    """
    memspec = f"#SBATCH --mem={args.memory}" if args.memory != "DefMemPerNode" else ""
    gpuspec = f"#SBATCH --gres={args.gres}" if args.gpu else ""
    nv_flag = "--nv \\" if args.gpu else "\\"
    env_file = f"env-{exp_name}.env"
    jobid = "-%j"

    # Write env vars one per line (supports values with commas e.g. TASKS_JSON)
    env_file_content = "\n".join(env_vars) + "\n"

    return dedent(f"""
#!/bin/bash
#SBATCH --job-name=armada-{exp_name}
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks={args.threads}
#SBATCH --cpus-per-task={args.cpus}
{memspec}
{gpuspec}
#SBATCH --time={args.time_limit}
#SBATCH --output=logs/armada-{exp_name}{jobid}.log
#SBATCH --error=logs/armada-{exp_name}{jobid}.err
#SBATCH --partition={args.partition}
#SBATCH -A {args.account}

echo "..............................................................."
echo "Experiment: {exp_name}"
echo "Node: $(hostname)"
echo "..............................................................."
module load apptainer
module load awscli
echo "..............................................................."
lscpu
echo "..............................................................."
mkdir -p {args.log_bind_host}
# Write env file at job start so SLURM_JOB_ID is available for WORKFLOW_ID
cat > {env_file} << 'ENVEOF'
{env_file_content}ENVEOF
time srun --exact --nodes {args.nodes} apptainer exec \\
    --env-file {env_file} \\
    --bind {args.log_bind_host}:{args.log_bind_container} \\
    --bind ${{HOME}}/.aws:${{HOME}}/.aws \\
    {nv_flag}
    --containall \\
    {args.docker_image} \\
    /rivanna/runArmada.sh
echo "..............................................................."
    """).strip(), env_file_content


def main():
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.log_bind_host, exist_ok=True)

    if args.smoke:
        # Single smoke test — mirrors ECS smoke test for direct comparison
        platform = "rivanna-gpu" if args.gpu else "rivanna"
        backend = "GCYLON" if args.gpu else "NUMPY"
        exp_name = f"{platform}_smoke_ws1"
        results_dir = f"results/{platform}/weak/"
        tasks_json = json.dumps(SMOKE_TASKS)

        env_vars = make_env_vars(
            args, tasks_json, exp_name, results_dir,
            backend=backend, dim=1024, threshold=0.85, world_size=1,
        )

        banner(f"SMOKE TEST: {exp_name}")
        script, env_file_content = make_slurm_script(args, exp_name, env_vars)
        filename = f"script-{exp_name}.slurm"
        writefile(filename, script)
        writefile(f"env-{exp_name}.env", env_file_content)
        print(script)

        if not args.dry_run and not debug:
            r = os.system(f"sbatch {filename}")
            if r == 0:
                Console.ok(f"Submitted: {filename}")
            else:
                Console.error(f"Failed to submit: {filename}")
        return

    # Full parameter sweep
    combinations = [
        (tc, th, dim, be)
        for tc in args.tasks
        for th in args.thresholds
        for dim in args.dimensions
        for be in args.backends
    ]

    platform = "rivanna-gpu" if args.gpu else "rivanna"
    scenario_file = (
        f"/cylon-armada/scripts/experiment/scenarios/{args.scenario}.json"
    )

    with open("submit.log", "w") as f:
        for counter, (tc, th, dim, be) in enumerate(combinations, 1):
            exp_name = f"{platform}_{args.scenario}_t{tc}_th{th}_d{dim}_{args.context_backend}_{be}"
            results_dir = f"results/{platform}/{args.scaling}/"

            # Load tasks from scenario file if it exists locally, else use placeholder
            scenario_local = os.path.join(
                os.path.dirname(__file__), "..", "..", "shared", "scripts",
                "experiment", "scenarios", f"{args.scenario}.json"
            )
            if os.path.exists(scenario_local):
                with open(scenario_local) as sf:
                    scenario_data = json.load(sf)
                tasks = scenario_data.get("tasks", [])[:tc]
                tasks_json = json.dumps(tasks)
            else:
                tasks_json = json.dumps([f"[{args.scenario} task {i+1}]" for i in range(tc)])

            env_vars = make_env_vars(
                args, tasks_json, exp_name, results_dir,
                backend=be, dim=dim, threshold=th, world_size=args.world_size,
            )

            banner(f"SLURM {exp_name} {counter}/{len(combinations)}")
            script, env_file_content = make_slurm_script(args, exp_name, env_vars)
            filename = f"script-{exp_name}.slurm"
            writefile(filename, script)
            writefile(f"env-{exp_name}.env", env_file_content)
            print(script)

            if not args.dry_run and not debug:
                r = os.system(f"sbatch {filename}")
                msg = (f"{counter} submitted: {exp_name}" if r == 0
                       else f"{counter} failed: {exp_name}")
                (Console.ok if r == 0 else Console.error)(msg)
                f.write(msg + "\n")

    print(f"\nGenerated {len(combinations)} SLURM scripts")
    if args.dry_run:
        print("Dry run — scripts generated but not submitted")


if __name__ == "__main__":
    main()