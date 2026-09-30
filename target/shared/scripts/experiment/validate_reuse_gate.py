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

"""Validate the reuse-validity gate against fresh inference, at the answer level.

reuse_policy.py's own docstring names the gap: its tolerance is bounded
against the model's own prediction residual, never against whether a reused
analysis actually agrees with what fresh inference would say for the query
galaxy. docs/Retrieve_Scaling_Findings_2026-09-22.md calls this the thing
"everything downstream depends on" and names it the top open item.

Method: replay the gate offline (experiment.reuse_replay, no LLM calls) to
find which (query, candidate) pairs it would accept as a reuse. Sample a
stratified set across the gate-error range — tight matches and near-boundary
ones — then, for each sampled pair, make two FRESH Bedrock calls (one for
the candidate's own prompt, standing in for "the analysis that got cached",
one for the query's own prompt, "what the query's own fresh inference would
say") and a third call asking an LLM judge whether the two reach the same
domain conclusion. Agreement rate at the answer level is what validates (or
doesn't) the gate — see reuse_policy.py for why redshift-residual agreement
was not enough.

Requires the dataset regenerate_reuse_gate_dataset.py produces:
    predictions.npy, true_redshifts.npy, magnitudes.npy, embeddings.npy,
    prompts.json — index-aligned, one row per galaxy.

Usage:
    python -m experiment.validate_reuse_gate --live --sample-size 40
"""

import argparse
import json
import os
import random
import re

import numpy as np

DATASET_FILES = ("predictions.npy", "true_redshifts.npy", "magnitudes.npy",
                  "embeddings.npy", "prompts.json")

DEFAULT_S3_PREFIX = "cylon-armada/results/exp_e_reuse_gate_validation/dataset/"
DEFAULT_RESULTS_BUCKET = "staylor.dev2"

JUDGE_SYSTEM_PROMPT = (
    "You are verifying whether two independent analyses of the same "
    "astronomical measurement reach the same substantive conclusion. You are "
    "not grading writing quality, style, or completeness \u2014 only whether "
    "they agree on the specific judgment below. Two analyses can use "
    "different words and still agree; they can use similar words and still "
    "disagree.\n\n"
    "Respond in exactly this format:\n"
    "VERDICT: AGREE or DISAGREE\n"
    "REASON: <one sentence>"
)

# Keyed by the template name task_generator.py would have used for the
# QUERY galaxy — the judge checks whether the candidate's cached answer is an
# acceptable stand-in for what the query actually asked.
DOMAIN_QUESTIONS = {
    "redshift_analysis": (
        "Is the redshift prediction accurate, and what does it imply for "
        "the galaxy's distance estimate?"
    ),
    "photometry_classification": (
        "Which bands (if any) are flagged as anomalous (|standardized "
        "value| > 2), and is the overall conclusion the same — anomalous "
        "or typical?"
    ),
    "outlier_analysis": (
        "Is the prediction error significant, and what are the likely "
        "causes?"
    ),
}


def template_for_index(idx, predictions, true_redshifts, outlier_threshold, index_offset=0):
    """Which task_generator.py template galaxy `idx` would have used.

    Mirrors generate_tasks_from_results()'s own branching exactly (residual
    vs. outlier_threshold, then index parity) rather than having that
    function report it, since it's a pure function of arrays already saved
    by regenerate_reuse_gate_dataset.py — no second source of truth needed.
    """
    residual = abs(float(predictions[idx]) - float(true_redshifts[idx]))
    if residual > outlier_threshold:
        return "outlier_analysis"
    if (index_offset + idx) % 2 == 0:
        return "redshift_analysis"
    return "photometry_classification"


def stratified_sample(accepted_pairs, sample_size, n_strata=4, seed=7):
    """Sample across the gate-error range, not purely at random.

    A pair whose gate_error sits near the tolerance boundary is the
    informative case for whether the tolerance is well-calibrated; a pure
    random sample would be dominated by the tight matches every corpus has
    far more of. Splits the accepted pairs into n_strata quantile buckets by
    gate_error and samples evenly across them, so the boundary is covered
    even though it's rare.
    """
    if not accepted_pairs:
        return []
    rng = random.Random(seed)
    with_error = [p for p in accepted_pairs if p[2] is not None]
    if not with_error:
        return rng.sample(accepted_pairs, min(sample_size, len(accepted_pairs)))

    # Split by INDEX, not by handing the tuple list to np.array_split
    # directly — that silently coerces each (int, int, float) tuple into a
    # numpy row, and rows compare with `in`/`remove` by ambiguous truth
    # value rather than identity, breaking the sampling below.
    ordered = sorted(with_error, key=lambda p: p[2])
    bucket_indices = np.array_split(np.arange(len(ordered)), n_strata)
    buckets = [[ordered[i] for i in idx] for idx in bucket_indices]
    per_bucket = max(1, sample_size // n_strata)
    sampled = []
    for bucket in buckets:
        bucket = list(bucket)
        if not bucket:
            continue
        take = min(per_bucket, len(bucket))
        sampled.extend(rng.sample(bucket, take))
    # Top up from the full pool if strata were small and we're under target.
    remaining = [p for p in with_error if p not in sampled]
    while len(sampled) < min(sample_size, len(with_error)) and remaining:
        pick = rng.choice(remaining)
        sampled.append(pick)
        remaining.remove(pick)
    return sampled[:sample_size]


def build_judge_prompt(domain_question, text_a, text_b):
    return (
        f"Judgment question: {domain_question}\n\n"
        f"Analysis A:\n{text_a}\n\n"
        f"Analysis B:\n{text_b}\n\n"
        "Do Analysis A and Analysis B reach the same conclusion on the "
        "judgment question above?"
    )


_VERDICT_RE = re.compile(r"VERDICT:\s*(AGREE|DISAGREE)", re.IGNORECASE)


def parse_verdict(judge_text):
    """AGREE/DISAGREE from the judge's response, or None if unparseable.

    None is a distinct outcome from DISAGREE — a judge call that didn't
    follow the format is a measurement failure, not evidence the gate is
    wrong, and must not be silently folded into either bucket.
    """
    match = _VERDICT_RE.search(judge_text or "")
    return match.group(1).upper() if match else None


def load_dataset(local_dir):
    predictions = np.load(os.path.join(local_dir, "predictions.npy"))
    true_redshifts = np.load(os.path.join(local_dir, "true_redshifts.npy"))
    magnitudes = np.load(os.path.join(local_dir, "magnitudes.npy"))
    embeddings = np.load(os.path.join(local_dir, "embeddings.npy"))
    with open(os.path.join(local_dir, "prompts.json")) as f:
        prompts = json.load(f)
    n = len(predictions)
    for name, arr in (("true_redshifts", true_redshifts), ("magnitudes", magnitudes),
                       ("embeddings", embeddings), ("prompts", prompts)):
        if len(arr) != n:
            raise ValueError(f"{name} has {len(arr)} rows, predictions has {n} — "
                              f"dataset files are not index-aligned")
    return predictions, true_redshifts, magnitudes, embeddings, prompts


def download_dataset(local_dir, bucket, s3_prefix):
    import boto3
    os.makedirs(local_dir, exist_ok=True)
    s3 = boto3.client("s3")
    for name in DATASET_FILES:
        s3.download_file(bucket, s3_prefix + name, os.path.join(local_dir, name))


def standardized_features(predictions, magnitudes):
    """z_pred plus the 5 magnitudes, z-scored together across the population.

    The magnitudes are already AstroMAE's standardized input features;
    predicted redshift is raw and on a different scale, so it needs the same
    treatment before the two are combined in one Euclidean distance — without
    it, whichever feature happens to have the larger raw spread would
    dominate the combined distance regardless of its actual relevance.
    """
    combined = np.column_stack([predictions, magnitudes])
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std == 0] = 1.0  # a constant column contributes nothing either way
    return (combined - mean) / std


def oracle_redshift_accuracy_features(predictions, true_redshifts):
    """z_pred plus the prediction residual |z_pred - z_true|, standardized.

    ORACLE ONLY — not a deployable gate. true_redshifts is the ground-truth
    label the model is being scored against; a real Retrieve decision only
    ever has z_pred at decision time, never the label used to grade it.

    Exists to test a specific hypothesis, not to ship: redshift_analysis's
    domain question is about PREDICTION ACCURACY, which depends on the
    residual, not on z_pred's absolute value. Two galaxies can share z_pred
    while one prediction is accurate and the other isn't — z_pred-only
    gating cannot see that. If gating on residual (however unrealistically)
    fixes the cross-galaxy agreement gap, that confirms the hypothesis and
    the next step is finding a deployable proxy for accuracy that doesn't
    require the label — this function is not that proxy.
    """
    residual = np.abs(predictions - true_redshifts)
    combined = np.column_stack([predictions, residual])
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std == 0] = 1.0
    return (combined - mean) / std


def fit_accuracy_proxy(predictions, magnitudes, true_redshifts, n_splits=5, seed=7):
    """Cross-validated, out-of-fold predicted residual for every galaxy.

    The deployable counterpart to the oracle: fits
    GradientBoostingRegressor(X=[z_pred, magnitudes] -> y=|z_pred-z_true|)
    with K-fold cross-validation, so every row's returned prediction comes
    from a model that never saw that row's own label during fitting — the
    same accuracy a frozen, once-trained proxy would give on a genuinely new
    query, using only what's available at decision time (z_pred, magnitudes).

    true_redshifts is used here only to build the OFFLINE TRAINING target,
    exactly the way any accuracy-proxy model would use labels once during
    fitting — AstroMAE itself was trained the same way. A deployed proxy
    would be this same fit, persisted, applied to a NEW query's [z_pred,
    magnitudes] with no label lookup at all; this function represents that
    offline fit-and-validate step, not the online decision itself.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold, cross_val_predict

    X = np.column_stack([predictions, magnitudes])
    y = np.abs(predictions - true_redshifts)
    model = GradientBoostingRegressor(random_state=seed)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(model, X, y, cv=cv)


def deployable_redshift_accuracy_features(predictions, magnitudes, true_redshifts,
                                           n_splits=5, seed=7):
    """z_pred plus the cross-validated PREDICTED residual, standardized —
    what oracle_redshift_accuracy_features would need to become to ship.
    """
    predicted_residual = fit_accuracy_proxy(predictions, magnitudes, true_redshifts,
                                             n_splits, seed)
    combined = np.column_stack([predictions, predicted_residual])
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std == 0] = 1.0
    return (combined - mean) / std


def find_tolerance_for_target_reuse_rate(embeddings, gate_values, threshold, epoch_batch_size,
                                          target_reuse_rate, tol_lo=1e-6, iterations=25):
    """Binary search the gate_tolerance whose reuse_rate is closest to target.

    Offline only (replay(), no LLM calls) — free to search exhaustively.
    reuse_rate is monotone non-decreasing in tolerance for a fixed candidate
    set (loosening the gate can only accept more, never fewer, of the pairs
    the cosine threshold already proposed), so bisection is valid. Matching
    the CURRENT reuse rate under a new gate shape is what makes the answer-
    level agreement comparison apples-to-apples: same measured benefit
    (reuse_rate), different question (is the accepted reuse actually valid).
    """
    from experiment.reuse_replay import SHARED, replay, totals

    def reuse_rate_at(tolerance):
        summaries = replay(embeddings, world_size=1, topology=SHARED, threshold=threshold,
                            epoch_batch_size=epoch_batch_size, gate_values=gate_values,
                            gate_tolerance=tolerance)
        return totals(summaries)["reuse_rate"]

    spread = np.ptp(gate_values, axis=0)
    tol_hi = float(np.linalg.norm(spread)) if gate_values.ndim > 1 else float(spread)
    tol_hi = max(tol_hi, tol_lo * 10)

    lo, hi = tol_lo, tol_hi
    best_tol, best_rate, best_gap = hi, reuse_rate_at(hi), None
    best_gap = abs(best_rate - target_reuse_rate)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        rate = reuse_rate_at(mid)
        gap = abs(rate - target_reuse_rate)
        if gap < best_gap:
            best_gap, best_tol, best_rate = gap, mid, rate
        if rate < target_reuse_rate:
            lo = mid
        else:
            hi = mid
    return best_tol, best_rate


def find_accepted_pairs(embeddings, gate_values, threshold, gate_tolerance, epoch_batch_size):
    from experiment.reuse_replay import SHARED, replay
    summaries = replay(embeddings, world_size=1, topology=SHARED, threshold=threshold,
                        epoch_batch_size=epoch_batch_size, gate_values=gate_values,
                        gate_tolerance=gate_tolerance, record_accepted_pairs=True)
    return summaries[0]["accepted_pairs"], summaries[0]


def run_validation(dataset, sample_size, threshold, gate_tolerance, epoch_batch_size,
                    seed, chain_executor, gate_values=None):
    predictions, true_redshifts, magnitudes, embeddings, prompts = dataset
    if gate_values is None:
        gate_values = predictions
    accepted_pairs, gate_summary = find_accepted_pairs(
        embeddings, gate_values, threshold, gate_tolerance, epoch_batch_size)
    sample = stratified_sample(accepted_pairs, sample_size, seed=seed)

    residuals = np.abs(predictions - true_redshifts)
    outlier_threshold = float(np.percentile(residuals, 90))

    rng = random.Random(seed)
    rows = []
    for query_idx, candidate_idx, gate_error in sample:
        response_candidate = chain_executor.execute(prompts[candidate_idx])
        response_query = chain_executor.execute(prompts[query_idx])

        template = template_for_index(query_idx, predictions, true_redshifts,
                                       outlier_threshold)
        domain_question = DOMAIN_QUESTIONS[template]

        # Randomize A/B position so the judge can't learn a position bias
        # from which side is consistently "the reused answer".
        swapped = rng.random() < 0.5
        text_a = response_query["response"] if swapped else response_candidate["response"]
        text_b = response_candidate["response"] if swapped else response_query["response"]

        judge_prompt = build_judge_prompt(domain_question, text_a, text_b)
        judge_result = chain_executor.execute(judge_prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        verdict = parse_verdict(judge_result["response"])

        rows.append({
            "query_idx": query_idx,
            "candidate_idx": candidate_idx,
            "gate_error": gate_error,
            "template": template,
            "domain_question": domain_question,
            "candidate_response": response_candidate["response"],
            "query_response": response_query["response"],
            "judge_raw": judge_result["response"],
            "verdict": verdict,
        })

    parsed = [r["verdict"] for r in rows if r["verdict"] is not None]
    agreement_rate = (parsed.count("AGREE") / len(parsed)) if parsed else None
    return {
        "gate_summary": {k: v for k, v in gate_summary.items() if k != "accepted_pairs"},
        "n_accepted_pairs": len(accepted_pairs),
        "n_sampled": len(rows),
        "n_parsed": len(parsed),
        "n_unparseable": len(rows) - len(parsed),
        "agreement_rate": agreement_rate,
        "rows": rows,
    }


def run_self_consistency_check(dataset, sample_size, seed, chain_executor):
    """Baseline ceiling: does the model agree with itself on the SAME prompt,
    called twice independently — no gate, no candidate matching at all.

    If three different gate designs all land at the same ~agreement rate
    regardless of how tight the match is, the next question is whether ANY
    gate could ever do better. This isolates LLM/task self-consistency from
    candidate-selection quality: a low ceiling here means no similarity
    signal could fix the reuse mechanism, because even the right answer
    doesn't reliably match itself; the fix would belong at the prompt/
    output-format level, not the gate.
    """
    predictions, true_redshifts, magnitudes, embeddings, prompts = dataset
    rng = random.Random(seed)
    sample_indices = rng.sample(range(len(prompts)), min(sample_size, len(prompts)))

    residuals = np.abs(predictions - true_redshifts)
    outlier_threshold = float(np.percentile(residuals, 90))

    rows = []
    for idx in sample_indices:
        response_1 = chain_executor.execute(prompts[idx])
        response_2 = chain_executor.execute(prompts[idx])

        template = template_for_index(idx, predictions, true_redshifts, outlier_threshold)
        domain_question = DOMAIN_QUESTIONS[template]

        swapped = rng.random() < 0.5
        text_a = response_2["response"] if swapped else response_1["response"]
        text_b = response_1["response"] if swapped else response_2["response"]

        judge_prompt = build_judge_prompt(domain_question, text_a, text_b)
        judge_result = chain_executor.execute(judge_prompt, system_prompt=JUDGE_SYSTEM_PROMPT)
        verdict = parse_verdict(judge_result["response"])

        rows.append({
            "galaxy_idx": idx,
            "template": template,
            "domain_question": domain_question,
            "response_1": response_1["response"],
            "response_2": response_2["response"],
            "judge_raw": judge_result["response"],
            "verdict": verdict,
        })

    parsed = [r["verdict"] for r in rows if r["verdict"] is not None]
    agreement_rate = (parsed.count("AGREE") / len(parsed)) if parsed else None
    return {
        "n_sampled": len(rows),
        "n_parsed": len(parsed),
        "n_unparseable": len(rows) - len(parsed),
        "agreement_rate": agreement_rate,
        "rows": rows,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-dir", default="/tmp/reuse_gate_dataset")
    parser.add_argument("--s3-bucket", default=DEFAULT_RESULTS_BUCKET)
    parser.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX)
    parser.add_argument("--skip-download", action="store_true",
                         help="reuse whatever is already in --local-dir")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.85,
                         help="matches SIMILARITY_THRESHOLD")
    parser.add_argument("--reuse-tolerance", type=float, default=0.0091,
                         help="matches fargate_cosmic_poc.py's --reuse-tolerance default, "
                              "the value the 146-stored/0.884-reuse finding used. Ignored "
                              "when --gate-features multi (the tolerance is searched instead)")
    parser.add_argument("--gate-features", choices=["redshift", "multi"], default="redshift",
                         help="redshift: gate on predicted redshift alone (today's deployed "
                              "policy). multi: gate on predicted redshift plus the 5 "
                              "standardized magnitudes via Euclidean distance, with the "
                              "tolerance searched to match --target-reuse-rate")
    parser.add_argument("--target-reuse-rate", type=float, default=0.884,
                         help="only used with --gate-features multi: search for the "
                              "tolerance whose reuse_rate is closest to this, so the "
                              "comparison against the redshift-only gate is at the same "
                              "measured reuse benefit")
    parser.add_argument("--epoch-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="/tmp/reuse_gate_validation_result.json")
    parser.add_argument("--mode", choices=["reuse-gate", "self-consistency"],
                         default="reuse-gate",
                         help="reuse-gate: validate gate-accepted reuses against fresh "
                              "inference (the default). self-consistency: baseline ceiling "
                              "— call the SAME prompt twice, no gate, no candidate matching "
                              "at all, to see whether any gate could ever do better")
    parser.add_argument("--live", action="store_true", required=True,
                         help="required — this makes real Bedrock calls "
                              "(2 x sample_size analysis calls + sample_size judge calls)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if not args.skip_download:
        print(f"downloading dataset from s3://{args.s3_bucket}/{args.s3_prefix}")
        download_dataset(args.local_dir, args.s3_bucket, args.s3_prefix)
    dataset = load_dataset(args.local_dir)
    print(f"loaded {len(dataset[0])} galaxies")

    from chain.executor import ChainExecutor
    chain_executor = ChainExecutor()

    if args.mode == "self-consistency":
        result = run_self_consistency_check(dataset, args.sample_size, args.seed,
                                             chain_executor)
        result["mode"] = "self-consistency"
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"sampled {result['n_sampled']}, parsed {result['n_parsed']}, "
              f"unparseable {result['n_unparseable']}")
        if result["agreement_rate"] is not None:
            print(f"self-consistency agreement rate: {result['agreement_rate']:.3f}")
        print(f"full results: {args.out}")
        return 0

    predictions, true_redshifts, magnitudes, embeddings, prompts = dataset
    if args.gate_features == "multi":
        gate_values = standardized_features(predictions, magnitudes)
        tolerance, achieved_rate = find_tolerance_for_target_reuse_rate(
            embeddings, gate_values, args.threshold, args.epoch_batch_size,
            args.target_reuse_rate)
        print(f"multi-feature gate: searched tolerance={tolerance:.4f}, "
              f"achieves reuse_rate={achieved_rate:.3f} (target {args.target_reuse_rate})")
    else:
        gate_values = predictions
        tolerance = args.reuse_tolerance

    result = run_validation(dataset, args.sample_size, args.threshold,
                             tolerance, args.epoch_batch_size, args.seed,
                             chain_executor, gate_values=gate_values)
    result["gate_features"] = args.gate_features
    result["gate_tolerance_used"] = tolerance

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)

    print(f"gate: {result['n_accepted_pairs']} accepted pairs out of "
          f"{result['gate_summary']['retrievals']} retrievals "
          f"(reuse_rate={result['gate_summary']['reuse_rate']:.3f})")
    print(f"sampled {result['n_sampled']}, parsed {result['n_parsed']}, "
          f"unparseable {result['n_unparseable']}")
    if result["agreement_rate"] is not None:
        print(f"answer-level agreement rate: {result['agreement_rate']:.3f}")
    print(f"full results (including response text for spot-checking): {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())