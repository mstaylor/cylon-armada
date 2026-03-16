"""Tests for AgentCoordinator — task preparation and result aggregation."""

import os
import sys
import base64
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))

from coordinator.agent_coordinator import _ndarray_to_b64, b64_to_ndarray


class TestBase64Encoding:
    def test_roundtrip(self):
        original = np.random.randn(256).astype(np.float32)
        b64 = _ndarray_to_b64(original)
        decoded = b64_to_ndarray(b64)
        np.testing.assert_array_almost_equal(original, decoded)

    def test_different_dimensions(self):
        for dim in [256, 512, 1024]:
            original = np.random.randn(dim).astype(np.float32)
            b64 = _ndarray_to_b64(original)
            decoded = b64_to_ndarray(b64)
            assert decoded.shape == (dim,)
            np.testing.assert_array_almost_equal(original, decoded)

    def test_is_valid_base64(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b64 = _ndarray_to_b64(arr)
        # Should not raise
        base64.b64decode(b64)