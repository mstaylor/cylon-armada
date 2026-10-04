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

"""Tests for the Arm C Phase 1 tolerance sweep's offline scoring.

Run: pytest tests/test_armc_tolerance_sweep.py -v
"""

import os
import sys

import numpy as np
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from armada.reuse_policy import redshift_validator
from experiment import armc_tolerance_sweep as sweep
from experiment.reuse_replay import SHARED, replay


def _labels(*pairs):
    return {sweep.sha(text): label for text, label in pairs}


class TestStated:
    def test_matches_exact_token(self):
        assert sweep.stated("predicted z = 0.095 here", 0.095, 3)

    def test_rejects_longer_number(self):
        assert not sweep.stated("value 0.0951", 0.095, 3)
        assert not sweep.stated("value 10.095", 0.095, 3)

    def test_none_text(self):
        assert not sweep.stated(None, 0.1, 3)


class TestExtractorParse:
    def test_parses_in_order(self):
        text = "ITEM 1: ACCURATE\nITEM 2: inaccurate\nITEM 3: UNCLEAR"
        assert sweep.parse_extractor_labels(text, 3) == [sweep.NORMAL, sweep.ANOMALOUS,
                                                         sweep.UNCLEAR]

    def test_missing_item_is_none(self):
        assert sweep.parse_extractor_labels("ITEM 2: ACCURATE", 2) == [None, sweep.NORMAL]

    def test_out_of_range_and_duplicate_ignored(self):
        text = "ITEM 1: ACCURATE\nITEM 1: INACCURATE\nITEM 9: ACCURATE"
        assert sweep.parse_extractor_labels(text, 1) == [sweep.NORMAL]


class TestAnswerVerdict:
    def test_photometry_parses_flag_line(self):
        label, verdict = sweep.answer_verdict(sweep.PHOTOMETRY, "x\nFLAGGED_BANDS: g, r", {})
        assert label == sweep.ANOMALOUS
        assert verdict == (sweep.PHOTOMETRY, frozenset({"g", "r"}))

    def test_photometry_unparseable_is_unclear(self):
        label, verdict = sweep.answer_verdict(sweep.PHOTOMETRY, "no marker", {})
        assert label == sweep.UNCLEAR and verdict == (sweep.PHOTOMETRY, None)

    def test_accuracy_uses_classifier_label(self):
        labels = _labels(("text", sweep.NORMAL))
        assert sweep.answer_verdict(sweep.REDSHIFT, "text", labels) == (
            sweep.NORMAL, ("accuracy", sweep.NORMAL))

    def test_accuracy_without_label_is_unscored(self):
        assert sweep.answer_verdict(sweep.OUTLIER, "unlabelled", {}) == (None, None)


class TestWilson:
    def test_empty(self):
        assert sweep.wilson(0, 0) == (None, None)

    def test_bounds_contain_estimate(self):
        lo, hi = sweep.wilson(8, 10)
        assert lo < 0.8 < hi and 0 <= lo and hi <= 1


class TestSplit:
    def test_stratified_halves(self):
        templates = [sweep.REDSHIFT] * 10 + [sweep.PHOTOMETRY] * 6 + [sweep.OUTLIER] * 3
        split = sweep.split_assignment(templates, seed=1)
        for template in sweep.TEMPLATES:
            members = [i for i, t in enumerate(templates) if t == template]
            fit = sum(split[i] == sweep.FIT for i in members)
            assert fit == len(members) // 2

    def test_deterministic(self):
        templates = [sweep.REDSHIFT] * 20
        assert sweep.split_assignment(templates, 3) == sweep.split_assignment(templates, 3)


def _toy_data(predictions, true_redshifts, magnitudes=None):
    n = len(predictions)
    return {
        "predictions": np.asarray(predictions, dtype=np.float64),
        "true_redshifts": np.asarray(true_redshifts, dtype=np.float64),
        "magnitudes": (np.zeros((n, 5)) if magnitudes is None
                       else np.asarray(magnitudes, dtype=np.float64)),
    }


class TestScorePairs:
    def test_agreeing_reuse_and_stated_source_redshift(self):
        data = _toy_data([0.100, 0.120], [0.101, 0.121])
        templates = [sweep.REDSHIFT, sweep.REDSHIFT]
        responses_a = {0: "z = 0.100 is accurate", 1: "z = 0.120 is accurate"}
        responses_b = dict(responses_a)
        labels = _labels((responses_a[0], sweep.NORMAL), (responses_a[1], sweep.NORMAL))
        rows = sweep.score_pairs([(1, 0)], [0, 1], data, templates, responses_a, responses_b,
                                 labels)
        row = rows[1]
        assert row["reused"] and row["agree"] and row["scored"]
        assert row["states_source_redshift"]
        assert row["stated_zpred_error"] == pytest.approx(0.02)
        assert not row["uninformative"]
        assert not rows[0]["reused"]
        assert rows[0]["self_agree"]

    def test_false_normal_counted(self):
        data = _toy_data([0.100, 0.120], [0.101, 0.200])
        templates = [sweep.REDSHIFT, sweep.OUTLIER]
        responses_a = {0: "accurate", 1: "significant error"}
        labels = _labels(("accurate", sweep.NORMAL), ("significant error", sweep.ANOMALOUS))
        rows = sweep.score_pairs([(1, 0)], [0, 1], data, templates, responses_a,
                                 dict(responses_a), labels)
        summary = sweep.summarize(list(rows.values()))
        assert summary["false_normal"] == 1
        assert summary["verdict_agreement"] == 0.0
        assert summary["fraction_uninformative"] == 1.0

    def test_photometry_query_is_served_synthesized_answer(self):
        magnitudes = [[0, 0, 0, 0, 2.5], [0, 0, 0, 0, 2.6]]
        data = _toy_data([0.1, 0.1], [0.1, 0.1], magnitudes)
        templates = [sweep.PHOTOMETRY, sweep.PHOTOMETRY]
        responses_a = {0: "FLAGGED_BANDS: z", 1: "FLAGGED_BANDS: NONE"}
        rows = sweep.score_pairs([(1, 0)], [0, 1], data, templates, responses_a,
                                 dict(responses_a), {})
        row = rows[1]
        assert row["synthesized"]
        assert row["served_correct"]
        assert not row["fresh_correct"]
        assert not row["agree"]
        assert not row["states_source_redshift"]

    def test_photometry_answer_served_to_accuracy_query_is_wrong_template(self):
        data = _toy_data([0.1, 0.1], [0.1, 0.1])
        templates = [sweep.PHOTOMETRY, sweep.REDSHIFT]
        responses_a = {0: "FLAGGED_BANDS: NONE", 1: "accurate"}
        labels = _labels(("accurate", sweep.NORMAL))
        rows = sweep.score_pairs([(1, 0)], [0, 1], data, templates, responses_a,
                                 dict(responses_a), labels)
        assert rows[1]["served_label"] == sweep.WRONG_TEMPLATE
        assert not rows[1]["agree"]


class TestValidatorReplay:
    def _identical(self, n):
        return np.ones((n, 4), dtype=np.float32)

    def test_validator_refuses_far_keys(self):
        summaries = replay(self._identical(2), world_size=1, topology=SHARED,
                           epoch_batch_size=1, reuse_validator=redshift_validator(0.01),
                           query_keys=[0.1, 0.5], stored_keys=[0.1, 0.5],
                           record_accepted_pairs=True)
        assert summaries[0]["cache_hits"] == 0
        assert summaries[0]["gate_rejections"] == 1

    def test_validator_accepts_near_keys(self):
        summaries = replay(self._identical(2), world_size=1, topology=SHARED,
                           epoch_batch_size=1, reuse_validator=redshift_validator(0.01),
                           query_keys=[0.1, 0.105], stored_keys=[0.1, 0.105],
                           record_accepted_pairs=True)
        assert summaries[0]["accepted_pairs"] == [(1, 0, None)]

    def test_missing_stored_key_refuses(self):
        summaries = replay(self._identical(2), world_size=1, topology=SHARED,
                           epoch_batch_size=1, reuse_validator=redshift_validator(0.01),
                           query_keys=[0.1, 0.1], stored_keys=[None, 0.1])
        assert summaries[0]["cache_hits"] == 0

    def test_walks_past_refused_best_match(self):
        embeddings = np.array([[1, 0], [0.9, 0.1], [0.95, 0.05]], dtype=np.float32)
        summaries = replay(embeddings, world_size=1, topology=SHARED, threshold=0.5,
                           epoch_batch_size=2, reuse_validator=redshift_validator(0.01),
                           query_keys=[0.1, 0.5, 0.5], stored_keys=[0.1, 0.5, 0.5],
                           record_accepted_pairs=True)
        assert summaries[0]["accepted_pairs"] == [(2, 1, None)]


class TestSelectTolerance:
    def test_picks_largest_reuse_within_margin(self):
        def summary(reuse, agree, ceiling):
            return {f"{sweep.CONTINUOUS}|{sweep.FIT}": {
                "reuse_rate": reuse, "verdict_agreement": agree,
                "self_agreement_on_reused": ceiling}}
        results = {"s": {"0.001": summary(0.2, 0.95, 0.96),
                         "0.01": summary(0.5, 0.93, 0.96),
                         "0.1": summary(0.8, 0.70, 0.96)}}
        assert sweep.select_tolerance(results, "s", [0.001, 0.01, 0.1], 0.03) == 0.01
        assert sweep.select_tolerance(results, "s", [0.001, 0.01, 0.1], 0.0) is None


class TestKappa:
    def test_perfect_agreement(self):
        confusion = {"a": {"a": 5, "b": 0}, "b": {"a": 0, "b": 5}}
        assert sweep.cohen_kappa(confusion) == pytest.approx(1.0)

    def test_constant_served_label_is_zero(self):
        confusion = {"a": {"a": 9, "b": 0}, "b": {"a": 1, "b": 0}}
        assert sweep.cohen_kappa(confusion) == pytest.approx(0.0)

    def test_empty(self):
        assert sweep.cohen_kappa({"a": {"a": 0}}) is None
