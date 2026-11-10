import os
import sys

import pyarrow as pa
import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from experiment.exp_b_payloads import PAYLOAD_KINDS, arrow_table, payload_table_builder  # noqa: E402


def _ipc_bytes(table):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _string_bytes(table):
    kv = table.column("kv_pairs").combine_chunks()
    return sum(len(k.encode()) + len(v.encode())
               for k, v in zip(kv.field("k").to_pylist(), kv.field("v").to_pylist()))


def test_float32_table_holds_msg_size_bytes():
    table = arrow_table("float32", 4096, 1.0)
    assert table.column_names == ["v"]
    assert table.schema.field("v").type == pa.float32()
    assert table.num_rows == 1024


def test_kv_pairs_table_uses_the_experiment_e_schema():
    kv_type = pa.struct([pa.field("k", pa.large_utf8()), pa.field("v", pa.large_utf8())])
    assert arrow_table("kv_pairs", 4096, 1.0).schema == pa.schema([pa.field("kv_pairs", kv_type)])


@pytest.mark.parametrize("msg_size", [8, 4096, 1048576])
def test_kv_pairs_string_bytes_reach_msg_size(msg_size):
    table = arrow_table("kv_pairs", msg_size, 1.0)
    total = _string_bytes(table)
    row_bytes = total / table.num_rows
    assert msg_size <= total < msg_size + row_bytes + 1


def test_smallest_kv_pairs_cell_still_has_a_row():
    assert arrow_table("kv_pairs", 8, 1.0).num_rows >= 1


@pytest.mark.parametrize("kind", PAYLOAD_KINDS)
def test_same_kind_and_size_give_identical_ipc_bytes(kind):
    assert _ipc_bytes(arrow_table(kind, 32768, 2.0)) == _ipc_bytes(arrow_table(kind, 32768, 2.0))


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="bogus"):
        arrow_table("bogus", 64, 1.0)


def test_builder_shapes_match_the_cylon_builder():
    build = payload_table_builder("kv_pairs")
    assert build("barrier", 0, None, 2, 0, 0) == (None, 0)
    assert build("allreduce", 8, None, 2, 1, 0) == (2.0, 8)
    table, size = build("allgather", 4096, None, 2, 0, 0)
    assert isinstance(table, pa.Table) and size == 4096
    shards, _ = build("scatter", 4096, None, 3, 0, 0)
    assert len(shards) == 3
    assert build("scatter", 4096, None, 3, 1, 0) == ([], 4096)