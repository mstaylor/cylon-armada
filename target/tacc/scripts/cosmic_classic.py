"""Classic Cosmic AI (Subexperiment A) on an HPC cluster, with files in place of S3.

The partition map, model load, inference loop and metrics are the Lambda code in
target/aws/scripts/terraform-cosmic-ai/lambda_src, imported unchanged. Only the storage
calls differ: payload.json, the partitions and the per-rank results are files.

  init       write <run-dir>/payload.json, as initializer.lambda_handler does. --partitions
             past the file count reuses the files cyclically, as the paper's 1 TB runs did
  worker     one rank's inference, as inference.py's __main__ does
  summarize  concatenate the rank results, as summarizer.lambda_handler does
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LAMBDA_SRC = os.path.normpath(
    os.path.join(_HERE, "..", "..", "aws", "scripts", "terraform-cosmic-ai", "lambda_src"))


def _lambda_modules(lambda_src=None, code_dir=None):
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    lambda_src = lambda_src or os.environ.get("COSMIC_LAMBDA_SRC", DEFAULT_LAMBDA_SRC)
    for path in (code_dir, lambda_src):
        if path and path not in sys.path:
            sys.path.insert(0, path)
    import inference
    import initializer
    return initializer, inference


def partition_names(data_dir):
    return sorted(name for name in os.listdir(data_dir) if name.endswith(".pt"))


def build_payload(data_dir, file_limit, world_size, batch_size, result_path, lambda_src=None, partitions=None):
    initializer, _ = _lambda_modules(lambda_src)
    names = partition_names(data_dir)
    if not names:
        raise ValueError(f"no .pt partitions in {data_dir}")
    names = names[:min(file_limit, len(names))]
    partitions = len(names) if partitions is None else partitions
    if partitions < 1:
        raise ValueError(f"partitions must be at least 1, got {partitions}")
    cycled = [names[index % len(names)] for index in range(partitions)]
    data_map = initializer.assign_partitions(cycled, world_size)
    return {
        "world_size": world_size,
        "file_limit": len(names),
        "partitions": partitions,
        "batch_size": batch_size,
        "result_path": result_path,
        "data_map": {str(rank): path for rank, path in data_map.items()},
    }


def _read_partition(data_dir, data_path, inference):
    import torch

    if isinstance(data_path, list):
        return inference.concatenate_data([_read_partition(data_dir, p, inference) for p in data_path])
    return torch.load(os.path.join(data_dir, data_path), weights_only=False)


def _write_result(result_path, rank, info):
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, f"{rank}.json"), "w") as f:
        json.dump(info, f)


def run_worker(payload_path, data_dir, model_path, code_dir, rank, device, lambda_src=None):
    from torch.utils.data import DataLoader

    _, inference = _lambda_modules(lambda_src, code_dir)
    process_start_ts = time.time()

    t_payload = time.time()
    with open(payload_path) as f:
        payload = json.load(f)
    payload_fetch_s = time.time() - t_payload

    batch_size = int(payload["batch_size"])
    result_path = payload["result_path"]
    data_path = payload["data_map"][str(rank)]

    t_scatter = time.time()
    data = _read_partition(data_dir, data_path, inference)
    scatter_s = time.time() - t_scatter
    dataloader = DataLoader(data, batch_size=batch_size)

    t_model = time.time()
    model = inference.load_model(model_path, device)
    model_load_s = time.time() - t_model

    info = inference.run_inference(model, dataloader, device, batch_size, rank, result_path, data_path)
    info.update({
        "code_fetch_s": None, "cold_start": None, "init_s": None,
        "payload_fetch_s": payload_fetch_s, "scatter_s": scatter_s, "model_load_s": model_load_s,
        "comm_init_s": 0.0, "barrier_s": 0.0, "publish_s": None, "total_s": None,
    })

    t_publish = time.time()
    _write_result(result_path, rank, info)
    info["publish_s"] = time.time() - t_publish
    info["total_s"] = time.time() - process_start_ts
    _write_result(result_path, rank, info)
    return info


class IncompleteRun(Exception):
    def __init__(self, run_dir, missing_ranks):
        self.run_dir = run_dir
        self.missing_ranks = missing_ranks
        super().__init__(f"incomplete run {run_dir}: missing ranks {missing_ranks}")


COMBINED_FILE = "combined_data.json"
METRICS_FILE = "aggregate_metrics.json"


def summarize(run_dir, world_size):
    t_aggregate_start = time.time()
    results = os.path.join(run_dir, "results")
    present = set(os.listdir(results)) if os.path.isdir(results) else set()
    missing = [rank for rank in range(world_size) if f"{rank}.json" not in present]
    if missing:
        raise IncompleteRun(run_dir, missing)

    all_data = []
    bytes_aggregated = 0
    ranks_aggregated = 0
    for name in sorted(present):
        if not name.endswith(".json") or name in (COMBINED_FILE, METRICS_FILE):
            continue
        with open(os.path.join(results, name), "rb") as f:
            body = f.read()
        bytes_aggregated += len(body)
        data = json.loads(body.decode("utf-8"))
        all_data.extend(data if isinstance(data, list) else [data])
        ranks_aggregated += 1

    with open(os.path.join(results, COMBINED_FILE), "w") as f:
        json.dump(all_data, f)
    metrics = {
        "aggregate_s": time.time() - t_aggregate_start,
        "ranks_aggregated": ranks_aggregated,
        "bytes_aggregated": bytes_aggregated,
    }
    with open(os.path.join(results, METRICS_FILE), "w") as f:
        json.dump(metrics, f)
    return metrics


def _rank_from_env():
    for key in ("RANK", "SLURM_PROCID"):
        if os.environ.get(key) is not None:
            return int(os.environ[key])
    raise SystemExit("worker needs --rank, RANK or SLURM_PROCID")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lambda-src", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--data-dir", required=True)
    init.add_argument("--file-limit", type=int, required=True)
    init.add_argument("--world-size", type=int, required=True)
    init.add_argument("--batch-size", type=int, required=True)
    init.add_argument("--run-dir", required=True)
    init.add_argument("--partitions", type=int, default=None)

    worker = sub.add_parser("worker")
    worker.add_argument("--run-dir", required=True)
    worker.add_argument("--data-dir", required=True)
    worker.add_argument("--model-path", required=True)
    worker.add_argument("--code-dir", required=True)
    worker.add_argument("--rank", type=int, default=None)
    worker.add_argument("--device", default="cpu")

    summary = sub.add_parser("summarize")
    summary.add_argument("--run-dir", required=True)
    summary.add_argument("--world-size", type=int, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "init":
        os.makedirs(args.run_dir, exist_ok=True)
        payload = build_payload(args.data_dir, args.file_limit, args.world_size, args.batch_size,
                                os.path.join(args.run_dir, "results"), args.lambda_src, args.partitions)
        with open(os.path.join(args.run_dir, "payload.json"), "w") as f:
            json.dump(payload, f, indent=4)
        return payload
    if args.command == "worker":
        rank = args.rank if args.rank is not None else _rank_from_env()
        return run_worker(os.path.join(args.run_dir, "payload.json"), args.data_dir, args.model_path,
                          args.code_dir, rank, args.device, args.lambda_src)
    if args.command == "summarize":
        try:
            return summarize(args.run_dir, args.world_size)
        except IncompleteRun as error:
            raise SystemExit(str(error))
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()