"""Arrow payload tables for Experiment B, shared by every channel.

Cylon and the Ray channels build their per-cell tables here, so a given
(kind, msg_size, fill) gives byte-identical Arrow data on both sides.

float32 is the original payload: one float32 column `v` holding msg_size bytes.
kv_pairs is Experiment E's exchange schema, struct<k: large_utf8, v: large_utf8>,
with rows added until the key and value bytes reach msg_size.
"""

import math
import os

import numpy as np
import pyarrow as pa

PAYLOAD_KINDS = ("float32", "kv_pairs")
SCALAR_BYTES = 8
KV_TYPE = pa.struct([pa.field("k", pa.large_utf8()), pa.field("v", pa.large_utf8())])


def _kv_value_bytes():
    return int(os.environ.get("EXPB_KV_VALUE_BYTES", "64"))


def _float32_table(msg_size, fill):
    values = np.full(max(1, msg_size // 4), fill, dtype=np.float32)
    return pa.table({"v": pa.array(values)})


def _kv_pairs_table(msg_size, fill):
    prefix = f"{fill:g}-"
    key_width = len(prefix) + 8
    value = "x" * _kv_value_bytes()
    rows = max(1, math.ceil(msg_size / (key_width + len(value))))
    keys = pa.array([f"{prefix}{i:08d}" for i in range(rows)], type=pa.large_utf8())
    values = pa.array([value] * rows, type=pa.large_utf8())
    column = pa.StructArray.from_arrays([keys, values], fields=list(KV_TYPE))
    return pa.table({"kv_pairs": column})


_TABLE_BUILDERS = {"float32": _float32_table, "kv_pairs": _kv_pairs_table}


def arrow_table(kind, msg_size, fill):
    """The payload table of the given kind holding at least msg_size bytes."""
    if kind not in _TABLE_BUILDERS:
        raise ValueError(f"unknown payload kind {kind!r}: expected one of {PAYLOAD_KINDS}")
    return _TABLE_BUILDERS[kind](msg_size, fill)


def payload_table_builder(kind):
    """A payload_builder for measure_collective that returns plain pyarrow tables.

    Shapes match build_cylon_payload: a table for broadcast/gather/allgather,
    a list of shards on the root for scatter/scatterv, a float scalar for
    reduce/allreduce, and (None, 0) for barrier.
    """
    if kind not in _TABLE_BUILDERS:
        raise ValueError(f"unknown payload kind {kind!r}: expected one of {PAYLOAD_KINDS}")

    def build(name, msg_size, ctx, world_size, rank, root):
        if name == "barrier":
            return (None, 0)
        if name in ("reduce", "allreduce"):
            return (float(rank + 1), SCALAR_BYTES)
        if name in ("broadcast", "gather", "allgather"):
            return (arrow_table(kind, msg_size, float(rank + 1)), msg_size)
        if name == "scatter":
            if rank == root:
                return ([arrow_table(kind, msg_size, float(i + 1)) for i in range(world_size)],
                        msg_size)
            return ([], msg_size)
        if name == "scatterv":
            base = max(1, max(1, msg_size // 4) // max(1, world_size))
            own_bytes = base * (rank + 1) * 4
            if rank == root:
                return ([arrow_table(kind, base * (i + 1) * 4, float(i + 1))
                         for i in range(world_size)], own_bytes)
            return ([], own_bytes)
        return None

    return build