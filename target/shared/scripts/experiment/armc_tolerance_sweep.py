# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Arm C Phase 1: the reuse gate's tolerance operating characteristic.

Implements spec section 4.4 of
docs/superpowers/specs/2026-09-25-cosmicai-lambda-dataplane-design.md. Each
galaxy's prompt is answered twice by fresh inference and both answers are
cached; each answer's verdict is classified once; every tolerance is then
scored offline by replaying the deployed gate over the cached answers.

Subcommands, run in order: prepare, generate, classify, score. Every
Bedrock-calling subcommand accepts --dry-run, appends each call to
ledger.jsonl, and refuses a plan that would push the ledger past --max-calls.

Run: python -m experiment.armc_tolerance_sweep --help
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np

from armada.reuse_policy import (
    _parse_flagged_bands,
    photometry_ground_truth_flags,
    redshift_validator,
    synthesize_photometry_response,
    verified_photometry_reuse_key,
)
from armada.run_cosmic_local import reuse_keys_for
from armada.run_metrics import is_throttle_error
from cosmic_ai.task_generator import BANDS, generate_tasks_from_results, template_for
from experiment.reuse_replay import SHARED, replay

DATASET_ARRAYS = ("predictions", "true_redshifts", "magnitudes", "embeddings")
RESPONSES_FILE = "responses.jsonl"
VERDICTS_FILE = "verdicts.jsonl"
LEDGER_FILE = "ledger.jsonl"
SCORE_FILE = "score.json"
TABLE_FILE = "tolerance_table.csv"

REDSHIFT = "redshift_analysis"
OUTLIER = "outlier_analysis"
PHOTOMETRY = "photometry_classification"
TEMPLATES = (REDSHIFT, OUTLIER, PHOTOMETRY)
ACCURACY_TEMPLATES = (REDSHIFT, OUTLIER)
POOLED = "all"
CONTINUOUS = "redshift_plus_outlier"

NORMAL = "normal"
ANOMALOUS = "anomalous"
UNCLEAR = "unclear"
WRONG_TEMPLATE = "wrong_template"
REFERENCE_LABELS = (NORMAL, ANOMALOUS, UNCLEAR)
SERVED_LABELS = (NORMAL, ANOMALOUS, UNCLEAR, WRONG_TEMPLATE)

PRODUCTION_SCHEME = "production"
ZPRED_SCHEME = "zpred_only"
SCHEMES = (PRODUCTION_SCHEME, ZPRED_SCHEME)
UNGATED = "ungated"

FIT = "fit"
REPORT = "report"

EXTRACTOR_MODEL_ENV = "VERDICT_EXTRACTOR_MODEL_ID"
CHARS_PER_TOKEN = 4

EXTRACTOR_SYSTEM_PROMPT = (
    "You label the overall conclusion of written analyses of one galaxy's "
    "photometric redshift prediction. For each numbered analysis decide what "
    "it concludes about the prediction error.\n"
    "ACCURATE: the prediction is accurate or reliable, or the error is small, "
    "acceptable, or not significant.\n"
    "INACCURATE: the prediction is inaccurate or unreliable, or the error is "
    "significant, large, or substantial.\n"
    "UNCLEAR: the analysis reaches no conclusion about the error, or "
    "concludes both ways without settling.\n"
    "Label only the analysis's own stated conclusion, never your own opinion "
    "of the numbers.\n"
    "Respond with exactly one line per analysis, in order, formatted as "
    "ITEM <number>: <LABEL>, and nothing else."
)
_EXTRACTOR_LINE_RE = re.compile(r"ITEM\s*(\d+)\s*:\s*(ACCURATE|INACCURATE|UNCLEAR)",
                                re.IGNORECASE)
_EXTRACTOR_LABELS = {"ACCURATE": NORMAL, "INACCURATE": ANOMALOUS, "UNCLEAR": UNCLEAR}


class BudgetExceeded(RuntimeError):
    pass


class ThrottleDetected(RuntimeError):
    pass


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record, lock):
    with lock:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())


def dataset_dir(out_dir):
    return os.path.join(out_dir, "dataset")


def load_dataset(out_dir):
    base = dataset_dir(out_dir)
    arrays = {name: np.load(os.path.join(base, f"{name}.npy")) for name in DATASET_ARRAYS}
    with open(os.path.join(base, "prompts.json")) as f:
        arrays["prompts"] = json.load(f)
    with open(os.path.join(base, "manifest.json")) as f:
        arrays["manifest"] = json.load(f)
    return arrays


def outlier_threshold_of(predictions, true_redshifts):
    return float(np.percentile(np.abs(predictions - true_redshifts), 90))


def templates_of(predictions, true_redshifts, outlier_threshold):
    residuals = np.abs(predictions - true_redshifts)
    return [template_for(i, float(residuals[i]), outlier_threshold, 0)
            for i in range(len(predictions))]


def ledger_totals(out_dir):
    records = read_jsonl(os.path.join(out_dir, LEDGER_FILE))
    totals = {"calls": len(records), "cost_usd": 0.0, "by_kind": {}}
    for record in records:
        kind = totals["by_kind"].setdefault(record["kind"], {
            "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            "throttles": 0})
        kind["calls"] += 1
        kind["input_tokens"] += record.get("input_tokens", 0)
        kind["output_tokens"] += record.get("output_tokens", 0)
        kind["cost_usd"] += record.get("cost_usd", 0.0)
        kind["throttles"] += record.get("throttles", 0)
        totals["cost_usd"] += record.get("cost_usd", 0.0)
    return totals


def check_budget(out_dir, planned_calls, max_calls):
    spent = ledger_totals(out_dir)["calls"]
    if spent + planned_calls > max_calls:
        raise BudgetExceeded(
            f"{spent} calls already in the ledger plus {planned_calls} planned exceeds "
            f"--max-calls {max_calls}")
    return spent


def resolve_pricing():
    from cost.bedrock_pricing import BedrockPricing
    return BedrockPricing.resolve(region=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))


def resolve_bedrock_config(model_override=None):
    from cost.bedrock_pricing import BedrockConfig
    config = BedrockConfig.resolve()
    if model_override:
        config = dataclasses.replace(config, llm_model_id=model_override)
    return config


class ThrottleWatch:
    """Counts throttled Bedrock responses, including ones botocore retries
    silently, by watching every attempt's parsed response."""

    def __init__(self, client):
        self.count = 0
        self._lock = threading.Lock()
        client.meta.events.register("after-call.bedrock-runtime", self._after_call)

    def _after_call(self, http_response=None, parsed=None, **_):
        status = getattr(http_response, "status_code", None)
        if status == 429 or is_throttle_error(SimpleNamespace(response=parsed or {})):
            with self._lock:
                self.count += 1


def estimate_tokens(text):
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def prepare(args):
    source = args.source_dir
    predictions = np.load(os.path.join(source, "predictions.npy"))
    true_redshifts = np.load(os.path.join(source, "true_redshifts.npy"))
    magnitudes = np.load(os.path.join(source, "magnitudes.npy"))
    embeddings = np.load(os.path.join(source, "embeddings.npy"))
    with open(os.path.join(source, "prompts.json")) as f:
        source_prompts = json.load(f)
    n = len(predictions)
    outlier_threshold = outlier_threshold_of(predictions, true_redshifts)
    prompts = generate_tasks_from_results(
        predictions, true_redshifts, magnitudes, max_tasks=n, seed=args.task_seed,
        outlier_threshold=outlier_threshold, index_offset=0)[:n]
    changed = [i for i in range(n) if prompts[i] != source_prompts[i]]
    unchanged = [i for i in range(n) if prompts[i] == source_prompts[i]]
    print(f"population {n}, prompts changed since the source dataset: {len(changed)}, "
          f"unchanged: {len(unchanged)}")
    planned = len(changed) + (1 if unchanged else 0)

    from context.embedding import EmbeddingService
    config = resolve_bedrock_config()
    if config.embedding_dimensions != embeddings.shape[1]:
        raise ValueError(f"resolved embedding_dimensions {config.embedding_dimensions} does not "
                         f"match the source embeddings' width {embeddings.shape[1]}")
    pricing = resolve_pricing()
    est_cost = sum(pricing.get_embedding_cost(config.embedding_model_id,
                                              estimate_tokens(prompts[i])) for i in changed)
    print(f"embedding calls planned: {planned} ({config.embedding_model_id}, "
          f"{config.embedding_dimensions} dims), estimated cost ${est_cost:.4f}")
    if args.dry_run:
        return 0
    os.makedirs(args.out_dir, exist_ok=True)
    check_budget(args.out_dir, planned, args.max_calls)

    embedder = EmbeddingService(config=config)
    ledger, lock = os.path.join(args.out_dir, LEDGER_FILE), threading.Lock()

    def embed(i, kind):
        vector, meta = embedder.embed(prompts[i])
        append_jsonl(ledger, {"kind": kind, "galaxy": i, "model_id": config.embedding_model_id,
                              "input_tokens": meta.get("token_count", 0),
                              "cost_usd": pricing.get_embedding_cost(
                                  config.embedding_model_id, meta.get("token_count", 0))}, lock)
        return np.asarray(vector, dtype=np.float32)

    if unchanged:
        probe = unchanged[0]
        fresh = embed(probe, "embed_probe")
        stored = embeddings[probe]
        cosine = float(fresh @ stored / (np.linalg.norm(fresh) * np.linalg.norm(stored)))
        print(f"probe galaxy {probe}: cosine(fresh, stored) = {cosine:.6f}")
        if cosine < args.probe_min_cosine:
            raise RuntimeError("the resolved embedding config does not reproduce the source "
                               "dataset's embeddings; refusing to mix them")
    new_embeddings = embeddings.astype(np.float32).copy()
    for count, i in enumerate(changed, 1):
        new_embeddings[i] = embed(i, "embed")
        if count % 100 == 0:
            print(f"embedded {count}/{len(changed)}", flush=True)

    base = dataset_dir(args.out_dir)
    os.makedirs(base, exist_ok=True)
    np.save(os.path.join(base, "predictions.npy"), predictions)
    np.save(os.path.join(base, "true_redshifts.npy"), true_redshifts)
    np.save(os.path.join(base, "magnitudes.npy"), magnitudes)
    np.save(os.path.join(base, "embeddings.npy"), new_embeddings)
    with open(os.path.join(base, "prompts.json"), "w") as f:
        json.dump(prompts, f)
    with open(os.path.join(base, "manifest.json"), "w") as f:
        json.dump({"source_dir": os.path.abspath(source), "n": n,
                   "outlier_threshold": outlier_threshold, "task_seed": args.task_seed,
                   "prompts_reembedded": changed,
                   "embedding_model_id": config.embedding_model_id,
                   "embedding_dimensions": config.embedding_dimensions}, f, indent=1)
    print(f"dataset written to {base}")
    return 0


def select_galaxies(n, galaxy_indices):
    if not galaxy_indices:
        return list(range(n))
    bad = [g for g in galaxy_indices if g < 0 or g >= n]
    if bad:
        raise ValueError(f"galaxy indices out of range: {bad}")
    return sorted(set(galaxy_indices))


def cached_responses(out_dir, model_id):
    cache = {}
    for record in read_jsonl(os.path.join(out_dir, RESPONSES_FILE)):
        if record["model_id"] == model_id:
            cache[(record["galaxy"], record["sample"], record["prompt_sha"])] = record
    return cache


def run_calls(jobs, call, workers, throttle_watch):
    """Runs call(job) over jobs on a thread pool, stopping at the first
    throttle seen either as an exception or by throttle_watch."""
    stop = threading.Event()
    completed, errors = 0, []

    def guarded(job):
        if stop.is_set():
            return None
        try:
            call(job)
        except Exception as exc:
            if is_throttle_error(exc):
                stop.set()
                raise ThrottleDetected(str(exc)) from exc
            raise
        if throttle_watch.count:
            stop.set()
        return job

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(guarded, job) for job in jobs]
        for future in as_completed(futures):
            try:
                if future.result() is not None:
                    completed += 1
            except ThrottleDetected as exc:
                errors.append(exc)
            except Exception as exc:
                stop.set()
                errors.append(exc)
            if completed and completed % 100 == 0:
                print(f"  {completed}/{len(jobs)} calls, "
                      f"{time.perf_counter() - started:.0f}s", flush=True)
    if throttle_watch.count or any(isinstance(e, ThrottleDetected) for e in errors):
        raise ThrottleDetected(
            f"throttling seen ({throttle_watch.count} throttled attempts); stopped after "
            f"{completed} of {len(jobs)} calls. Cached results are kept; rerun resumes.")
    if errors:
        raise errors[0]
    return completed


def generate(args):
    data = load_dataset(args.out_dir)
    prompts = data["prompts"]
    config = resolve_bedrock_config()
    model_id = config.llm_model_id
    galaxies = select_galaxies(len(prompts), args.galaxy_indices)
    cache = cached_responses(args.out_dir, model_id)
    jobs = [(g, s) for s in range(args.samples) for g in galaxies
            if (g, s, sha(prompts[g])) not in cache]
    pricing = resolve_pricing()
    est_output = args.output_token_estimate
    est_cost = sum(pricing.get_llm_cost(model_id, estimate_tokens(prompts[g]), est_output)
                   for g, _ in jobs)
    print(f"generation: model {model_id}, {len(galaxies)} galaxies x {args.samples} samples, "
          f"{len(jobs)} uncached calls; estimated cost ${est_cost:.3f} at "
          f"{est_output} output tokens per call")
    if args.dry_run or not jobs:
        return 0
    check_budget(args.out_dir, len(jobs), args.max_calls)

    from chain.executor import ChainExecutor
    executor = ChainExecutor(config=config)
    watch = ThrottleWatch(executor.llm.client)
    responses_path = os.path.join(args.out_dir, RESPONSES_FILE)
    ledger_path = os.path.join(args.out_dir, LEDGER_FILE)
    lock = threading.Lock()

    def call(job):
        galaxy, sample = job
        result = executor.execute(prompts[galaxy])
        cost = pricing.get_llm_cost(model_id, result["input_tokens"], result["output_tokens"])
        append_jsonl(responses_path, {
            "galaxy": galaxy, "sample": sample, "prompt_sha": sha(prompts[galaxy]),
            "model_id": model_id, "response": result["response"],
            "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
            "latency_ms": result["latency_ms"]}, lock)
        append_jsonl(ledger_path, {"kind": "generate", "galaxy": galaxy, "sample": sample,
                                   "model_id": model_id,
                                   "input_tokens": result["input_tokens"],
                                   "output_tokens": result["output_tokens"],
                                   "cost_usd": cost}, lock)

    done = run_calls(jobs, call, args.workers, watch)
    print(f"generated {done} responses")
    return 0


def response_table(out_dir, prompts, model_id):
    table = {}
    for (galaxy, sample, prompt_sha), record in cached_responses(out_dir, model_id).items():
        if prompt_sha == sha(prompts[galaxy]):
            table[(galaxy, sample)] = record["response"]
    return table


def cached_verdicts(out_dir, extractor_model_id):
    cache = {}
    for record in read_jsonl(os.path.join(out_dir, VERDICTS_FILE)):
        if record["extractor_model_id"] == extractor_model_id:
            cache[record["response_sha"]] = record["label"]
    return cache


def build_extractor_prompt(texts):
    parts = [f"ANALYSIS {i}:\n<<<\n{text}\n>>>" for i, text in enumerate(texts, 1)]
    return "\n\n".join(parts) + f"\n\nLabel all {len(texts)} analyses."


def parse_extractor_labels(text, expected):
    labels = {}
    for number, label in _EXTRACTOR_LINE_RE.findall(text or ""):
        position = int(number)
        if 1 <= position <= expected and position not in labels:
            labels[position] = _EXTRACTOR_LABELS[label.upper()]
    return [labels.get(i) for i in range(1, expected + 1)]


def resolve_extractor_model(args):
    return args.extractor_model_id or os.environ.get(EXTRACTOR_MODEL_ENV) or \
        resolve_bedrock_config().llm_model_id


def classify(args):
    data = load_dataset(args.out_dir)
    prompts = data["prompts"]
    templates = templates_of(data["predictions"], data["true_redshifts"],
                             data["manifest"]["outlier_threshold"])
    generator_model = resolve_bedrock_config().llm_model_id
    extractor_model = resolve_extractor_model(args)
    responses = response_table(args.out_dir, prompts, generator_model)
    cache = cached_verdicts(args.out_dir, extractor_model)
    pending, seen = [], set()
    for (galaxy, sample), text in sorted(responses.items()):
        if templates[galaxy] not in ACCURACY_TEMPLATES:
            continue
        key = sha(text)
        if key in cache or key in seen:
            continue
        seen.add(key)
        pending.append((key, text))
    batches = [pending[i:i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
    pricing = resolve_pricing()
    est_cost = sum(pricing.get_llm_cost(
        extractor_model,
        estimate_tokens(EXTRACTOR_SYSTEM_PROMPT + build_extractor_prompt([t for _, t in b])),
        12 * len(b)) for b in batches)
    print(f"classification: extractor {extractor_model}, {len(pending)} unlabelled answers in "
          f"{len(batches)} batched calls of up to {args.batch_size}; "
          f"estimated cost ${est_cost:.3f}")
    if args.dry_run or not batches:
        return 0
    check_budget(args.out_dir, len(batches), args.max_calls)

    from chain.executor import ChainExecutor
    executor = ChainExecutor(config=resolve_bedrock_config(extractor_model))
    watch = ThrottleWatch(executor.llm.client)
    verdicts_path = os.path.join(args.out_dir, VERDICTS_FILE)
    ledger_path = os.path.join(args.out_dir, LEDGER_FILE)
    lock = threading.Lock()
    unresolved = []

    def call(batch):
        result = executor.execute(build_extractor_prompt([t for _, t in batch]),
                                  system_prompt=EXTRACTOR_SYSTEM_PROMPT)
        append_jsonl(ledger_path, {
            "kind": "classify", "items": len(batch), "model_id": extractor_model,
            "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
            "cost_usd": pricing.get_llm_cost(extractor_model, result["input_tokens"],
                                             result["output_tokens"])}, lock)
        for (key, _), label in zip(batch, parse_extractor_labels(result["response"], len(batch))):
            if label is None:
                with lock:
                    unresolved.append(key)
                continue
            append_jsonl(verdicts_path, {"response_sha": key, "label": label,
                                         "extractor_model_id": extractor_model,
                                         "batch_size": len(batch)}, lock)

    run_calls(batches, call, args.workers, watch)
    print(f"classified {len(pending) - len(unresolved)} answers; "
          f"{len(unresolved)} unparsed in their batch (rerun classify to retry them)")
    return 0


def stated(text, value, decimals):
    if text is None or value is None:
        return False
    token = f"{value:.{decimals}f}"
    pattern = r"(?<![\d.])" + re.escape(token) + r"(?!\d)"
    return re.search(pattern, text) is not None


def galaxy_numbers(prediction, true_redshift, magnitudes):
    numbers = [(float(prediction), 3), (float(true_redshift), 3),
               (abs(float(prediction) - float(true_redshift)), 4)]
    numbers.extend((float(v), 2) for v in magnitudes)
    return numbers


def answer_verdict(template, text, labels):
    """(label, comparable verdict) for an answer to `template`'s question."""
    if template == PHOTOMETRY:
        flags = _parse_flagged_bands(text)
        if flags is None:
            return UNCLEAR, (PHOTOMETRY, None)
        return (ANOMALOUS if flags else NORMAL), (PHOTOMETRY, flags)
    label = labels.get(sha(text)) if text is not None else None
    if label is None:
        return None, None
    return label, ("accuracy", label)


def wilson(successes, n, z=1.96):
    if n == 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def split_assignment(templates, seed):
    rng = np.random.default_rng(seed)
    split = {}
    for template in TEMPLATES:
        members = [i for i, t in enumerate(templates) if t == template]
        rng.shuffle(members)
        half = len(members) // 2
        for position, galaxy in enumerate(members):
            split[galaxy] = FIT if position < half else REPORT
    return split


def keys_for_scheme(scheme, data, templates, responses_a):
    predictions, true_redshifts = data["predictions"], data["true_redshifts"]
    magnitudes = data["magnitudes"]
    n = len(predictions)
    if scheme == ZPRED_SCHEME:
        keys = [float(p) for p in predictions]
        return keys, list(keys)
    residuals = np.abs(predictions - true_redshifts)
    query_keys = reuse_keys_for(predictions, true_redshifts, residuals, magnitudes,
                                data["manifest"]["outlier_threshold"], 0, n)
    stored_keys = list(query_keys)
    for i in range(n):
        if templates[i] == PHOTOMETRY:
            stored_keys[i] = verified_photometry_reuse_key(responses_a.get(i), magnitudes[i])
    return query_keys, stored_keys


def served_answer(query, candidate, templates, magnitudes, responses_a):
    """What production Reason returns on a hit: a synthesized answer for a
    photometry_classification query, else the candidate's cached answer."""
    if templates[query] == PHOTOMETRY:
        synthesized = synthesize_photometry_response(magnitudes[query])
        if synthesized is not None:
            return synthesized, templates[query], True
    return responses_a[candidate], templates[candidate], False


def score_pairs(pairs, population, data, templates, responses_a, responses_b, labels):
    predictions, true_redshifts = data["predictions"], data["true_redshifts"]
    magnitudes = data["magnitudes"]
    rows = {}
    for query in population:
        rows[query] = {"galaxy": query, "template": templates[query], "reused": False}
    for query, candidate in pairs:
        text, source_template, synthesized = served_answer(query, candidate, templates,
                                                           magnitudes, responses_a)
        ref_label, ref_verdict = answer_verdict(templates[query], responses_a[query], labels)
        if source_template == PHOTOMETRY and templates[query] != PHOTOMETRY:
            served_label, served_verdict = WRONG_TEMPLATE, (WRONG_TEMPLATE,)
        elif templates[query] == PHOTOMETRY and not synthesized and \
                source_template != PHOTOMETRY:
            served_label, served_verdict = WRONG_TEMPLATE, (WRONG_TEMPLATE,)
        else:
            served_label, served_verdict = answer_verdict(source_template, text, labels)
        source = query if synthesized else candidate
        numbers = galaxy_numbers(predictions[source], true_redshifts[source], magnitudes[source])
        informative = any(stated(text, v, d) for v, d in numbers)
        zpred_src, zpred_q = float(predictions[candidate]), float(predictions[query])
        ztrue_src, ztrue_q = float(true_redshifts[candidate]), float(true_redshifts[query])
        states_src_zpred = not synthesized and stated(text, zpred_src, 3)
        states_src_ztrue = not synthesized and stated(text, ztrue_src, 3)
        differs_zpred = f"{zpred_src:.3f}" != f"{zpred_q:.3f}"
        differs_ztrue = f"{ztrue_src:.3f}" != f"{ztrue_q:.3f}"
        rows[query].update({
            "reused": True, "candidate": candidate, "source_template": source_template,
            "synthesized": synthesized, "cross_template": source_template != templates[query],
            "ref_label": ref_label, "served_label": served_label,
            "agree": (ref_verdict is not None and served_verdict is not None
                      and ref_verdict == served_verdict),
            "scored": ref_verdict is not None and served_verdict is not None,
            "uninformative": not informative,
            "states_source_redshift": ((states_src_zpred and differs_zpred)
                                       or (states_src_ztrue and differs_ztrue)),
            "stated_zpred_error": abs(zpred_src - zpred_q) if states_src_zpred else None,
            "stated_ztrue_error": abs(ztrue_src - ztrue_q) if states_src_ztrue else None,
        })
    for query, row in rows.items():
        if templates[query] == PHOTOMETRY:
            truth = (PHOTOMETRY, photometry_ground_truth_flags(magnitudes[query]))
            row["fresh_correct"] = answer_verdict(PHOTOMETRY, responses_a[query], labels)[1] == truth
            if row["reused"]:
                text, source_template, _ = served_answer(query, row["candidate"], templates,
                                                         magnitudes, responses_a)
                row["served_correct"] = (source_template == PHOTOMETRY and
                                         answer_verdict(PHOTOMETRY, text, labels)[1] == truth)
        ref_label, ref_verdict = answer_verdict(templates[query], responses_a[query], labels)
        repeat_label, repeat_verdict = answer_verdict(templates[query], responses_b.get(query),
                                                      labels)
        row["self_scored"] = ref_verdict is not None and repeat_verdict is not None
        row["self_agree"] = row["self_scored"] and ref_verdict == repeat_verdict
        row["self_ref_label"], row["self_repeat_label"] = ref_label, repeat_label
    return rows


def summarize(rows):
    n = len(rows)
    reused = [r for r in rows if r["reused"]]
    scored = [r for r in reused if r["scored"]]
    agree = sum(r["agree"] for r in scored)
    confusion = {ref: {served: 0 for served in SERVED_LABELS} for ref in REFERENCE_LABELS}
    for r in scored:
        confusion[r["ref_label"]][r["served_label"]] += 1
    ref_anomalous = sum(confusion[ANOMALOUS].values())
    ref_normal = sum(confusion[NORMAL].values())
    zpred_errors = [r["stated_zpred_error"] for r in reused if r["stated_zpred_error"] is not None]
    ztrue_errors = [r["stated_ztrue_error"] for r in reused if r["stated_ztrue_error"] is not None]
    self_rows = [r for r in rows if r["self_scored"]]
    reused_self = [r for r in reused if r["self_scored"]]
    self_confusion = {ref: {served: 0 for served in SERVED_LABELS} for ref in REFERENCE_LABELS}
    for r in reused_self:
        if r["self_ref_label"] in self_confusion and r["self_repeat_label"] in SERVED_LABELS:
            self_confusion[r["self_ref_label"]][r["self_repeat_label"]] += 1
    self_anomalous = sum(self_confusion[ANOMALOUS].values())
    lo, hi = wilson(agree, len(scored))
    return {
        "n_queries": n,
        "n_reused": len(reused),
        "reuse_rate": len(reused) / n if n else None,
        "n_scored": len(scored),
        "verdict_agreement": agree / len(scored) if scored else None,
        "verdict_agreement_ci95": [lo, hi],
        "confusion": confusion,
        "false_normal": confusion[ANOMALOUS][NORMAL],
        "false_normal_rate": confusion[ANOMALOUS][NORMAL] / ref_anomalous if ref_anomalous else None,
        "kappa": cohen_kappa(confusion),
        "self_confusion_on_reused": self_confusion,
        "self_kappa_on_reused": cohen_kappa(self_confusion),
        "self_false_normal_rate_on_reused": (self_confusion[ANOMALOUS][NORMAL] / self_anomalous
                                             if self_anomalous else None),
        "false_outlier": confusion[NORMAL][ANOMALOUS],
        "false_outlier_rate": confusion[NORMAL][ANOMALOUS] / ref_normal if ref_normal else None,
        "anomalous_set_mismatch": sum(1 for r in scored if r["ref_label"] == ANOMALOUS
                                      and r["served_label"] == ANOMALOUS and not r["agree"]),
        "cross_template_fraction": (sum(r["cross_template"] for r in reused) / len(reused)
                                    if reused else None),
        "synthesized_fraction": (sum(r["synthesized"] for r in reused) / len(reused)
                                 if reused else None),
        "stated_redshift_error": distribution(zpred_errors),
        "stated_true_redshift_error": distribution(ztrue_errors),
        "fraction_stating_source_redshift": (sum(r["states_source_redshift"] for r in reused)
                                             / len(reused) if reused else None),
        "fraction_uninformative": (sum(r["uninformative"] for r in reused) / len(reused)
                                   if reused else None),
        "fresh_correct_vs_ground_truth": fraction(rows, "fresh_correct"),
        "served_correct_vs_ground_truth": fraction(reused, "served_correct"),
        "self_agreement": (sum(r["self_agree"] for r in self_rows) / len(self_rows)
                           if self_rows else None),
        "self_agreement_n": len(self_rows),
        "self_agreement_on_reused": (sum(r["self_agree"] for r in reused_self) / len(reused_self)
                                     if reused_self else None),
    }


def cohen_kappa(confusion):
    """Chance-corrected agreement over a reference-by-served label matrix."""
    total = sum(sum(row.values()) for row in confusion.values())
    if total == 0:
        return None
    observed = sum(confusion[label].get(label, 0) for label in confusion) / total
    expected = sum(
        (sum(confusion[label].values()) / total)
        * (sum(row.get(label, 0) for row in confusion.values()) / total)
        for label in confusion)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def fraction(rows, field):
    present = [r[field] for r in rows if field in r]
    return sum(present) / len(present) if present else None


def distribution(values):
    if not values:
        return {"n": 0, "median": None, "mean": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"n": int(arr.size), "median": float(np.median(arr)), "mean": float(arr.mean()),
            "p90": float(np.percentile(arr, 90)), "max": float(arr.max())}


def template_groups(template):
    groups = [template, POOLED]
    if template in ACCURACY_TEMPLATES:
        groups.append(CONTINUOUS)
    return groups


def summarize_by_group(rows, split):
    grouped = {}
    for row in rows.values():
        for group in template_groups(row["template"]):
            for part in ("full", split[row["galaxy"]]):
                grouped.setdefault((group, part), []).append(row)
    return {f"{group}|{part}": summarize(members) for (group, part), members in grouped.items()}


def replay_pairs(data, population, scheme, tolerance, query_keys, stored_keys, args):
    embeddings = data["embeddings"][population]
    local_query = [query_keys[g] for g in population]
    local_stored = [stored_keys[g] for g in population]
    kwargs = dict(world_size=args.world_size, topology=SHARED, threshold=args.threshold,
                  epoch_batch_size=args.epoch_batch_size, top_k=args.top_k,
                  record_accepted_pairs=True)
    if tolerance is not None:
        kwargs.update(reuse_validator=redshift_validator(tolerance), query_keys=local_query,
                      stored_keys=local_stored)
    summaries = replay(embeddings, **kwargs)
    pairs = [(population[q], population[c]) for s in summaries for q, c, _ in s["accepted_pairs"]]
    return pairs


def select_tolerance(results, scheme, tolerances, margin, group=CONTINUOUS):
    """Largest-reuse tolerance whose fit-split verdict agreement on `group`
    is within `margin` of the fit-split self-agreement on the same reused
    queries."""
    qualifying = []
    for tolerance in tolerances:
        summary = results[scheme][tolerance_label(tolerance)].get(f"{group}|{FIT}")
        if not summary or summary["verdict_agreement"] is None:
            continue
        ceiling = summary["self_agreement_on_reused"]
        if ceiling is None:
            continue
        if summary["verdict_agreement"] >= ceiling - margin:
            qualifying.append((summary["reuse_rate"], tolerance))
    if not qualifying:
        return None
    return max(qualifying)[1]


def tolerance_label(tolerance):
    return UNGATED if tolerance is None else f"{tolerance:g}"


def score(args):
    data = load_dataset(args.out_dir)
    prompts = data["prompts"]
    n = len(prompts)
    templates = templates_of(data["predictions"], data["true_redshifts"],
                             data["manifest"]["outlier_threshold"])
    generator_model = resolve_bedrock_config().llm_model_id
    extractor_model = resolve_extractor_model(args)
    table = response_table(args.out_dir, prompts, generator_model)
    responses_a = {g: t for (g, s), t in table.items() if s == 0}
    responses_b = {g: t for (g, s), t in table.items() if s == 1}
    population = [g for g in range(n) if g in responses_a and g in responses_b]
    labels = cached_verdicts(args.out_dir, extractor_model)
    missing = [g for g in population if templates[g] in ACCURACY_TEMPLATES and any(
        sha(r[g]) not in labels for r in (responses_a, responses_b))]
    print(f"scoring {len(population)} of {n} galaxies with both samples cached; "
          f"{len(missing)} accuracy-template galaxies lack a verdict label")
    if not population:
        return 1
    split = split_assignment(templates, args.split_seed)
    tolerances = list(args.tolerances) + [None]
    results = {}
    for scheme in SCHEMES:
        query_keys, stored_keys = keys_for_scheme(scheme, data, templates, responses_a)
        results[scheme] = {}
        for tolerance in tolerances:
            pairs = replay_pairs(data, population, scheme, tolerance, query_keys,
                                 stored_keys, args)
            rows = score_pairs(pairs, population, data, templates, responses_a, responses_b,
                               labels)
            results[scheme][tolerance_label(tolerance)] = summarize_by_group(rows, split)
            pooled = results[scheme][tolerance_label(tolerance)][f"{POOLED}|full"]
            print(f"  {scheme:>10} tol={tolerance_label(tolerance):>8} "
                  f"reuse={pooled['reuse_rate']:.3f} agree={pooled['verdict_agreement']}")
    selection = {scheme: {
        "selected_tolerance": select_tolerance(results, scheme, args.tolerances,
                                               args.agreement_margin),
        "rule": (f"largest reuse_rate among gated tolerances whose fit-split verdict agreement "
                 f"over {CONTINUOUS} is at least the fit-split self-agreement on the same "
                 f"reused queries minus {args.agreement_margin}")}
        for scheme in SCHEMES}
    ledger = ledger_totals(args.out_dir)
    output = {
        "population": len(population), "n_total": n,
        "generator_model_id": generator_model, "extractor_model_id": extractor_model,
        "threshold": args.threshold, "world_size": args.world_size,
        "epoch_batch_size": args.epoch_batch_size, "top_k": args.top_k,
        "split_seed": args.split_seed, "tolerances": [tolerance_label(t) for t in tolerances],
        "template_counts": {t: sum(1 for g in population if templates[g] == t)
                            for t in TEMPLATES},
        "missing_labels": len(missing), "selection": selection, "ledger": ledger,
        "results": results,
    }
    with open(os.path.join(args.out_dir, SCORE_FILE), "w") as f:
        json.dump(output, f, indent=1, default=lambda o: sorted(o) if isinstance(o, frozenset)
                  else str(o))
    write_table(output, os.path.join(args.out_dir, TABLE_FILE))
    print(json.dumps(selection, indent=1))
    print(f"ledger: {ledger['calls']} calls, ${ledger['cost_usd']:.4f}")
    print(f"wrote {os.path.join(args.out_dir, SCORE_FILE)} and {TABLE_FILE}")
    return 0


TABLE_COLUMNS = ("scheme", "tolerance", "group", "split", "n_queries", "n_reused", "reuse_rate",
                 "n_scored", "verdict_agreement", "ci95_lo", "ci95_hi",
                 "self_agreement", "self_agreement_on_reused", "false_normal",
                 "false_normal_rate", "false_outlier", "false_outlier_rate",
                 "anomalous_set_mismatch", "stated_redshift_error_median",
                 "stated_redshift_error_p90", "stated_redshift_error_n",
                 "stated_true_redshift_error_median", "fraction_stating_source_redshift",
                 "fraction_uninformative", "cross_template_fraction", "synthesized_fraction",
                 "fresh_correct_vs_ground_truth", "served_correct_vs_ground_truth",
                 "kappa", "self_kappa_on_reused", "self_false_normal_rate_on_reused")


def write_table(output, path):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TABLE_COLUMNS)
        for scheme, by_tolerance in output["results"].items():
            for tolerance, by_group in by_tolerance.items():
                for key, s in sorted(by_group.items()):
                    group, part = key.split("|")
                    writer.writerow([
                        scheme, tolerance, group, part, s["n_queries"], s["n_reused"],
                        s["reuse_rate"], s["n_scored"], s["verdict_agreement"],
                        s["verdict_agreement_ci95"][0], s["verdict_agreement_ci95"][1],
                        s["self_agreement"], s["self_agreement_on_reused"], s["false_normal"],
                        s["false_normal_rate"], s["false_outlier"], s["false_outlier_rate"],
                        s["anomalous_set_mismatch"], s["stated_redshift_error"]["median"],
                        s["stated_redshift_error"]["p90"], s["stated_redshift_error"]["n"],
                        s["stated_true_redshift_error"]["median"],
                        s["fraction_stating_source_redshift"], s["fraction_uninformative"],
                        s["cross_template_fraction"], s["synthesized_fraction"],
                        s["fresh_correct_vs_ground_truth"], s["served_correct_vs_ground_truth"],
                        s["kappa"], s["self_kappa_on_reused"],
                        s["self_false_normal_rate_on_reused"]])


def env_float(name, fallback):
    raw = os.environ.get(name)
    return float(raw) if raw is not None else fallback


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--out-dir", required=True)
        p.add_argument("--max-calls", type=int,
                       default=int(os.environ.get("ARMC_MAX_BEDROCK_CALLS", 3 * 1253)),
                       help="refuse any plan that would push ledger.jsonl past this many "
                            "Bedrock calls (env ARMC_MAX_BEDROCK_CALLS)")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--workers", type=int,
                       default=int(os.environ.get("ARMC_BEDROCK_WORKERS", 4)))

    p = sub.add_parser("prepare", help="regenerate prompts and re-embed the changed ones")
    common(p)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--task-seed", type=int, default=42)
    p.add_argument("--probe-min-cosine", type=float, default=0.999)
    p.set_defaults(fn=prepare)

    p = sub.add_parser("generate", help="fresh answers, cached, resumable")
    common(p)
    p.add_argument("--samples", type=int, default=2,
                   help="answers per galaxy; sample 0 is the cached fresh answer, sample 1 "
                        "the repeat behind the self-agreement ceiling")
    p.add_argument("--galaxy-indices", type=int, nargs="*")
    p.add_argument("--output-token-estimate", type=int,
                   default=int(os.environ.get("ARMC_OUTPUT_TOKEN_ESTIMATE", 700)))
    p.set_defaults(fn=generate)

    p = sub.add_parser("classify", help="verdict label per accuracy-template answer")
    common(p)
    p.add_argument("--batch-size", type=int,
                   default=int(os.environ.get("ARMC_CLASSIFY_BATCH_SIZE", 8)))
    p.add_argument("--extractor-model-id", default=None,
                   help=f"default env {EXTRACTOR_MODEL_ENV}, else BEDROCK_LLM_MODEL_ID")
    p.set_defaults(fn=classify)

    p = sub.add_parser("score", help="offline replay and metrics at every tolerance")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tolerances", type=float, nargs="+",
                   default=[0.0005, 0.001, 0.002, 0.005, 0.0091, 0.02, 0.05])
    p.add_argument("--threshold", type=float,
                   default=env_float("SIMILARITY_THRESHOLD", 0.85))
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--epoch-batch-size", type=int,
                   default=int(os.environ.get("EPOCH_BATCH_SIZE", 4)))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=20261002)
    p.add_argument("--agreement-margin", type=float,
                   default=env_float("ARMC_AGREEMENT_MARGIN", 0.02))
    p.add_argument("--extractor-model-id", default=None)
    p.set_defaults(fn=score)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (BudgetExceeded, ThrottleDetected) as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
