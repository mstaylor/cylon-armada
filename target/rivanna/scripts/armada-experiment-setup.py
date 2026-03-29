"""Generate SLURM scripts for cylon-armada experiments on Rivanna.

Follows the cylon experiment-setup-apptainer.py pattern:
  1. Generates SLURM scripts from parameter combinations
  2. Each script runs inside the Apptainer container
  3. Uses --env to pass configuration, --bind for output

Usage:
    # CPU experiments (standard partition)
    python armada-experiment-setup.py \
        -d cylon-armada.sif \
        -r1 localhost -t 4 -n 1 -c 10 \
        --tasks 4 8 16 --thresholds 0.8 --dimensions 256 \
        --runs 3 --scenario hydrology \
        -l1 /scratch/$USER/armada-results -l2 /output \
        -p2 standard

    # GPU experiments (bii-gpu partition)
    python armada-experiment-setup.py \
        -d cylon-armada-gpu.sif \
        -r1 localhost -t 1 -n 1 -c 10 \
        --tasks 4 8 16 --thresholds 0.8 --dimensions 256 \
        --runs 3 --scenario hydrology \
        --gpu \
        -l1 /scratch/$USER/armada-results -l2 /output \
        -p2 bii-gpu
"""

import os
import argparse
from textwrap import dedent
from cloudmesh.common.util import writefile
from cloudmesh.common.util import banner
from cloudmesh.common.console import Console

debug = False


def build_parser():
    parser = argparse.ArgumentParser(description="cylon-armada Rivanna experiment setup")

    # SLURM configuration
    parser.add_argument("-n", dest="nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("-t", dest="threads", type=int, default=1, help="Tasks per node")
    parser.add_argument("-c", dest="cpus", type=int, default=10, help="CPUs per task")
    parser.add_argument("-p2", dest="partition", type=str, default="standard")
    parser.add_argument("-m", dest="memory", type=str, default="DefMemPerNode")
    parser.add_argument("--time", dest="time_limit", type=str, default="4:00:00")
    parser.add_argument("--gpu", action="store_true", help="Request GPU resources")
    parser.add_argument("--gres", type=str, default="gpu:1", help="GPU resource spec")

    # Container
    parser.add_argument("-d", dest="docker_image", type=str, required=True,
                        help="Apptainer .sif image path")

    # Redis
    parser.add_argument("-r1", dest="redis_host", type=str, default="localhost")
    parser.add_argument("-p1", dest="redis_port", type=int, default=6379)

    # Experiment parameters
    parser.add_argument("--tasks", type=int, nargs="+", default=[4, 8, 16],
                        help="Task counts to test")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.8],
                        help="Similarity thresholds")
    parser.add_argument("--dimensions", type=int, nargs="+", default=[256],
                        help="Embedding dimensions")
    parser.add_argument("--backends", type=str, nargs="+", default=["NUMPY"],
                        help="SIMD backends")
    parser.add_argument("--context-backend", type=str, default="redis",
                        choices=["redis", "cylon"])
    parser.add_argument("--runs", type=int, default=3, help="Runs per config for std dev")
    parser.add_argument("--scenario", type=str, default="hydrology",
                        help="Scenario name (hydrology, epidemiology, seismology, mixed_scientific)")
    parser.add_argument("--sampling", type=str, default="sequential",
                        choices=["stratified", "sequential", "random"])

    # Bind mounts
    parser.add_argument("-l1", dest="log_bind_host", type=str, required=True,
                        help="Host directory for results output")
    parser.add_argument("-l2", dest="log_bind_container", type=str, default="/output",
                        help="Container directory for results output")

    # Submit control
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts without submitting")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    counter = 0
    total_jobs = 0

    # Build experiment combinations
    combinations = []
    for tc in args.tasks:
        for th in args.thresholds:
            for dim in args.dimensions:
                for be in args.backends:
                    combinations.append((tc, th, dim, be))

    total_jobs = len(combinations)

    memspec = ""
    if args.memory != "DefMemPerNode":
        memspec = f"#SBATCH --mem={args.memory}"

    gpuspec = ""
    if args.gpu:
        gpuspec = f"#SBATCH --gres={args.gres}"

    jobid = "-%j"

    os.makedirs(args.log_bind_host, exist_ok=True)
    f = open("submit.log", "w")

    for tc, th, dim, be in combinations:
        counter += 1

        exp_name = f"t{tc}_th{th}_d{dim}_{args.context_backend}_{be}"
        scenario_file = f"/cylon-armada/scripts/experiment/scenarios/{args.scenario}.json"

        env_vars = [
            f"REDIS_HOST={args.redis_host}",
            f"REDIS_PORT={args.redis_port}",
            f"CONTEXT_BACKEND={args.context_backend}",
            f"TASKS={tc}",
            f"THRESHOLDS={th}",
            f"DIMENSIONS={dim}",
            f"BACKENDS={be}",
            f"RUNS={args.runs}",
            f"SAMPLING={args.sampling}",
            f"SCENARIO_FILE={scenario_file}",
        ]
        env_vars_str = ",".join(env_vars)

        banner(f"SLURM {exp_name} {counter}/{total_jobs}")

        script = dedent(f"""
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
#SBATCH -A bii_dsc_community
echo "..............................................................."
echo "Experiment: {exp_name}"
echo "Node: $(hostname)"
echo "..............................................................."
module load apptainer
echo "..............................................................."
lscpu
echo "..............................................................."
time srun --exact --nodes {args.nodes} apptainer run \\
    --env {env_vars_str} \\
    --bind {args.log_bind_host}:{args.log_bind_container} \\
    {"--nv" if args.gpu else ""} \\
    --containall \\
    {args.docker_image}
echo "..............................................................."
        """).strip()

        print(script)
        filename = f"script-{exp_name}.slurm"
        writefile(filename, script)

        if not args.dry_run and not debug:
            r = os.system(f"sbatch {filename}")
            if r == 0:
                msg = f"{counter} submitted: {exp_name}"
                Console.ok(msg)
            else:
                msg = f"{counter} failed: {exp_name}"
                Console.error(msg)
            f.writelines([msg, "\n"])

    f.close()
    print(f"\nGenerated {counter} SLURM scripts")
    if args.dry_run:
        print("Dry run — scripts generated but not submitted")


if __name__ == "__main__":
    main()