"""Tests for ContextManager — configuration-driven backend selection."""

import os
import sys
import json
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'target', 'shared', 'scripts'))


class TestContextManagerRedisBackend:
    """Tests using the 'redis' backend (numpy+Redis).

    Default config: persist_to_redis=True, dynamo_table=None.
    Redis is the primary persistence layer; DynamoDB is disabled.
    """

    @patch('context.manager._create_redis_backend')
    def test_store_context_persists_to_redis(self, mock_create_redis):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        embedding = np.random.randn(256).astype(np.float32)

        cm.store_context(
            workflow_id="wf-1",
            task_description="test task",
            embedding=embedding,
            response="test response",
            cost_metadata={"model_id": "test-model", "input_tokens": 100},
        )

        # Embedding pipeline (setex + sadd + expire) should execute
        mock_pipe.execute.assert_called_once()
        # Metadata JSON should be written via setex
        mock_redis.setex.assert_called_once()
        setex_key = mock_redis.setex.call_args[0][0]
        assert setex_key.startswith("context:")

    @patch('context.manager._create_redis_backend')
    def test_store_context_no_dynamo_by_default(self, mock_create_redis):
        """DynamoDB should NOT be called when dynamo_table is None (default)."""
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        assert cm._table is None

        cm.store_context(
            workflow_id="wf-1",
            task_description="test task",
            embedding=np.random.randn(256).astype(np.float32),
            response="resp",
            cost_metadata={},
        )
        # No DynamoDB put_item
        assert cm._dynamo is None

    @patch('context.manager._create_redis_backend')
    def test_store_context_with_dynamo(self, mock_create_redis):
        """DynamoDB put_item called when dynamo_table is provided."""
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        with patch('context.manager.boto3') as mock_boto3:
            mock_dynamo_table = MagicMock()
            mock_boto3.resource.return_value.Table.return_value = mock_dynamo_table

            cm = ContextManager(backend="redis", dynamo_table="my-table")
            cm.store_context(
                workflow_id="wf-1",
                task_description="test task",
                embedding=np.random.randn(256).astype(np.float32),
                response="resp",
                cost_metadata={},
            )

        mock_dynamo_table.put_item.assert_called_once()

    @patch('context.manager._create_redis_backend')
    def test_get_context_from_redis(self, mock_create_redis):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({
            "response": "cached response",
            "metadata": {},
        }).encode()
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        result = cm.get_context("ctx-123")

        assert result is not None
        assert result["response"] == "cached response"
        assert result["source"] == "redis"

    @patch('context.manager._create_redis_backend')
    def test_get_all_embeddings(self, mock_create_redis):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        mock_redis.smembers.return_value = {b"ctx-1", b"ctx-2"}
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [emb1.tobytes(), emb2.tobytes()]
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        results = cm.get_all_embeddings(workflow_id="wf-1")

        assert len(results) == 2
        for ctx_id, embedding in results:
            assert isinstance(ctx_id, str)
            assert isinstance(embedding, np.ndarray)
            assert embedding.dtype == np.float32

    @patch('context.manager._create_redis_backend')
    def test_search_redis(self, mock_create_redis):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        mock_redis.smembers.return_value = {b"ctx-1", b"ctx-2"}
        mock_pipe = MagicMock()
        mock_pipe.execute.return_value = [emb1.tobytes(), emb2.tobytes()]
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        query = np.array([0.99, 0.1, 0.0], dtype=np.float32)
        results = cm.search(query, threshold=0.5, workflow_id="wf-1")

        assert len(results) >= 1
        assert results[0]["similarity"] > 0.5
        if len(results) > 1:
            assert results[0]["similarity"] >= results[1]["similarity"]

    @patch('context.manager._create_redis_backend')
    def test_increment_reuse_count_updates_redis(self, mock_create_redis):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({
            "response": "resp",
            "reuse_count": 2,
        }).encode()
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        cm.increment_reuse_count("ctx-123", "wf-1")

        # Redis setex should be called with incremented count
        mock_redis.setex.assert_called_once()
        written = json.loads(mock_redis.setex.call_args[0][2])
        assert written["reuse_count"] == 3

    @patch('context.manager._create_redis_backend')
    def test_get_workflow_contexts_from_redis(self, mock_create_redis):
        from context.manager import ContextManager

        ctx_data = {"response": "resp", "workflow_id": "wf-1", "task_description": "t"}
        mock_redis = MagicMock()
        mock_redis.smembers.return_value = {b"ctx-1"}
        mock_redis.get.return_value = json.dumps(ctx_data).encode()
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis")
        results = cm.get_workflow_contexts("wf-1")

        assert len(results) == 1
        assert results[0]["context_id"] == "ctx-1"

    @patch('context.manager._create_redis_backend')
    def test_in_memory_fallback_when_no_persistence(self, mock_create_redis):
        """When persist_to_redis=False and no dynamo_table, contexts go to _memory."""
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_pipe = MagicMock()
        mock_redis.pipeline.return_value = mock_pipe
        mock_create_redis.return_value = {"client": mock_redis}

        cm = ContextManager(backend="redis", persist_to_redis=False)
        embedding = np.random.randn(256).astype(np.float32)

        ctx_id = cm.store_context(
            workflow_id="wf-1",
            task_description="in-memory task",
            embedding=embedding,
            response="resp",
            cost_metadata={},
        )

        assert ctx_id in cm._memory
        assert cm._memory[ctx_id]["task_description"] == "in-memory task"
        # No metadata JSON written to Redis
        mock_redis.setex.assert_not_called()


class TestContextManagerValidation:
    def test_invalid_backend_raises(self):
        from context.manager import ContextManager
        with pytest.raises(ValueError, match="Invalid context_backend"):
            ContextManager(backend="invalid")

    @patch('context.manager._create_redis_backend')
    def test_from_config(self, mock_create_redis):
        from context.manager import ContextManager
        from cost.bedrock_pricing import BedrockConfig

        mock_create_redis.return_value = {"client": MagicMock()}

        config = BedrockConfig(
            context_backend="redis",
            embedding_dimensions=512,
        )
        cm = ContextManager.from_config(config)
        assert cm._backend_name == "redis"
        assert cm._embedding_dim == 512

class TestContextTableSnapshot:
    """The Arrow IPC snapshot to Redis on every cylon store.

    It is cold-start persistence, not part of the data plane, and it has to be
    switchable off: it serializes the WHOLE table on every store, so its cost
    is O(stores^2) across a run, and it sits inside the region Experiment E
    times on the arm whose claim is zero-copy.
    """

    @staticmethod
    def _backend(save_fn):
        return {"table": MagicMock(), "redis_addr": "h:1", "redis_ttl": 60,
                "has_redis": True, "save_fn": save_fn, "load_fn": None}

    @staticmethod
    def _store(cm):
        cm.store_context(workflow_id="wf", task_description="t",
                         embedding=np.zeros(8, dtype=np.float32), response="r",
                         cost_metadata={})

    @patch('context.manager._create_cylon_backend')
    def test_the_context_table_snapshot_can_be_disabled(self, mock_create_cylon):
        from context.manager import ContextManager

        save_fn = MagicMock()
        mock_create_cylon.return_value = self._backend(save_fn)

        self._store(ContextManager(backend="cylon", embedding_dim=8,
                                   snapshot_context_table=False))

        save_fn.assert_not_called()

    @patch('context.manager._create_cylon_backend')
    def test_the_snapshot_is_on_by_default(self, mock_create_cylon):
        from context.manager import ContextManager

        save_fn = MagicMock()
        mock_create_cylon.return_value = self._backend(save_fn)

        self._store(ContextManager(backend="cylon", embedding_dim=8))

        save_fn.assert_called_once()

    @patch('context.manager._create_cylon_backend')
    def test_the_environment_can_turn_the_snapshot_off(self, mock_create_cylon, monkeypatch):
        """The sweep sets CONTEXT_TABLE_SNAPSHOT rather than threading a
        constructor argument through every worker entry point."""
        from context.manager import ContextManager

        monkeypatch.setenv("CONTEXT_TABLE_SNAPSHOT", "0")
        save_fn = MagicMock()
        mock_create_cylon.return_value = self._backend(save_fn)

        self._store(ContextManager(backend="cylon", embedding_dim=8))

        save_fn.assert_not_called()

    @patch('context.manager._create_cylon_backend')
    def test_an_explicit_argument_beats_the_environment(self, mock_create_cylon, monkeypatch):
        """Env -> event -> config -> default: an explicit setting is the caller
        being specific and must not be overridden by the ambient environment."""
        from context.manager import ContextManager

        monkeypatch.setenv("CONTEXT_TABLE_SNAPSHOT", "0")
        save_fn = MagicMock()
        mock_create_cylon.return_value = self._backend(save_fn)

        self._store(ContextManager(backend="cylon", embedding_dim=8,
                                   snapshot_context_table=True))

        save_fn.assert_called_once()

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "False", "no", "off", "", " 0 "])
    @patch('context.manager._create_cylon_backend')
    def test_every_spelling_of_off_turns_the_snapshot_off(self, mock_create_cylon,
                                                          monkeypatch, value):
        """An operator who types FALSE, no or off means off. Silently reading
        those as on would leave a serialization step inside the timed region of
        the arm whose claim is zero-copy, and nothing would report it."""
        from context.manager import ContextManager

        monkeypatch.setenv("CONTEXT_TABLE_SNAPSHOT", value)
        save_fn = MagicMock()
        mock_create_cylon.return_value = self._backend(save_fn)

        self._store(ContextManager(backend="cylon", embedding_dim=8))

        save_fn.assert_not_called()

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    @patch('context.manager._create_cylon_backend')
    def test_every_spelling_of_on_turns_the_snapshot_on(self, mock_create_cylon,
                                                        monkeypatch, value):
        from context.manager import ContextManager

        monkeypatch.setenv("CONTEXT_TABLE_SNAPSHOT", value)
        save_fn = MagicMock()
        mock_create_cylon.return_value = self._backend(save_fn)

        self._store(ContextManager(backend="cylon", embedding_dim=8))

        save_fn.assert_called_once()

    @patch('context.manager._create_cylon_backend')
    def test_an_unrecognized_value_fails_fast_instead_of_guessing(self, mock_create_cylon,
                                                                  monkeypatch):
        """Fail fast at the boundary: a typo must not resolve to either state
        by accident, because both are plausible and neither is reported."""
        from context.manager import ContextManager

        monkeypatch.setenv("CONTEXT_TABLE_SNAPSHOT", "maybe")
        mock_create_cylon.return_value = self._backend(MagicMock())

        with pytest.raises(ValueError, match="CONTEXT_TABLE_SNAPSHOT"):
            ContextManager(backend="cylon", embedding_dim=8)


class TestReuseKey:
    """The value a reuse-validity predicate compares.

    The manager does not interpret it; it persists it and gives it back. The
    Arrow ContextTable has no column for it, so the cylon path depends on the
    process-local map — which is the case the Armada arm actually runs.
    """

    @patch('context.manager._create_redis_backend')
    def test_a_stored_reuse_key_reaches_the_redis_metadata(self, mock_create_redis):
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_create_redis.return_value = {"client": mock_redis}
        cm = ContextManager(backend="redis", embedding_dim=8)

        cm.store_context(workflow_id="wf", task_description="t",
                         embedding=np.zeros(8, dtype=np.float32), response="r",
                         cost_metadata={}, context_id="c-1", reuse_key=0.42)

        stored = json.loads(mock_redis.setex.call_args_list[-1][0][2])
        assert stored["reuse_key"] == 0.42

    @patch('context.manager._create_redis_backend')
    def test_no_reuse_key_leaves_the_record_unchanged(self, mock_create_redis):
        """Every existing caller passes none; the stored record must not gain a
        field for them, or prior contexts stop comparing equal."""
        from context.manager import ContextManager

        mock_redis = MagicMock()
        mock_create_redis.return_value = {"client": mock_redis}
        cm = ContextManager(backend="redis", embedding_dim=8)

        cm.store_context(workflow_id="wf", task_description="t",
                         embedding=np.zeros(8, dtype=np.float32), response="r",
                         cost_metadata={}, context_id="c-1")

        assert "reuse_key" not in json.loads(mock_redis.setex.call_args_list[-1][0][2])

    @patch('context.manager._create_cylon_backend')
    def test_the_cylon_path_returns_the_key_the_arrow_table_cannot_hold(self, mock_create_cylon):
        """The ContextTable schema is fixed C++-side, so get_context's cylon
        branch would otherwise return a record with no key and silently disable
        gating on exactly the arm whose gate matters."""
        from context.manager import ContextManager

        row = MagicMock()
        row.column.return_value.__getitem__.return_value.as_py.return_value = "x"
        table = MagicMock()
        table.get.return_value = row
        mock_create_cylon.return_value = {
            "table": table, "redis_addr": "h:1", "redis_ttl": 60,
            "has_redis": False, "save_fn": None, "load_fn": None}

        cm = ContextManager(backend="cylon", embedding_dim=8, persist_to_redis=False)
        cm.store_context(workflow_id="wf", task_description="t",
                         embedding=np.zeros(8, dtype=np.float32), response="r",
                         cost_metadata={}, context_id="c-1", reuse_key=0.31)

        assert cm.get_context("c-1")["reuse_key"] == 0.31

    @patch('context.manager._create_cylon_backend')
    def test_a_context_stored_without_a_key_reports_none(self, mock_create_cylon):
        """None means unverifiable, and a predicate must refuse it rather than
        treat the absence as permission."""
        from context.manager import ContextManager

        row = MagicMock()
        row.column.return_value.__getitem__.return_value.as_py.return_value = "x"
        table = MagicMock()
        table.get.return_value = row
        mock_create_cylon.return_value = {
            "table": table, "redis_addr": "h:1", "redis_ttl": 60,
            "has_redis": False, "save_fn": None, "load_fn": None}

        cm = ContextManager(backend="cylon", embedding_dim=8, persist_to_redis=False)
        cm.store_context(workflow_id="wf", task_description="t",
                         embedding=np.zeros(8, dtype=np.float32), response="r",
                         cost_metadata={}, context_id="c-2")

        assert cm.get_context("c-2")["reuse_key"] is None

    @patch('context.manager._create_redis_backend')
    def test_the_in_memory_fallback_returns_the_key(self, mock_create_redis):
        """Reachable config: redis search backend with persistence off. The key
        is in the process map but the in-memory branch dropped it, so gating
        silently degraded to ungated for every row."""
        from context.manager import ContextManager

        mock_create_redis.return_value = {"client": MagicMock()}
        cm = ContextManager(backend="redis", embedding_dim=8, persist_to_redis=False)
        cm.store_context(workflow_id="wf", task_description="t",
                         embedding=np.zeros(8, dtype=np.float32), response="r",
                         cost_metadata={}, context_id="c-1", reuse_key=0.27)

        assert cm.get_context("c-1")["reuse_key"] == 0.27

    @patch('context.manager._create_redis_backend')
    def test_clearing_a_workflow_prunes_its_reuse_keys(self, mock_create_redis):
        """The map is keyed by context id with no workflow, so nothing else
        would ever remove an entry and it would grow for the life of the run."""
        from context.manager import ContextManager

        mock_create_redis.return_value = {"client": MagicMock()}
        cm = ContextManager(backend="redis", embedding_dim=8, persist_to_redis=False)
        cm.store_context(workflow_id="wf", task_description="t",
                         embedding=np.zeros(8, dtype=np.float32), response="r",
                         cost_metadata={}, context_id="c-1", reuse_key=0.27)

        cm.clear_workflow("wf")

        assert "c-1" not in cm._reuse_keys
