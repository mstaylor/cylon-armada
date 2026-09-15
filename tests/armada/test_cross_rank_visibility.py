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

"""The property Experiment E exists to measure.

Arm A's claim is that a context published on one rank becomes visible to the
others through the collective. Nothing else in the suite asserts it: every
other test checks mechanism — that a collective fired, that counts agree — and
a pipeline whose AllGather moved only acknowledgement booleans passed all of
them, through eight task reviews. This test fails in that case and passes only
when contexts really cross.

Two ranks are simulated in one process, each with its own ContextManager, wired
to a bridge that exchanges tables between them as a real allgather would.

Run: pytest tests/armada/test_cross_rank_visibility.py -v
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from armada.cosmic_workflow import build_cosmic_workflow
from armada.executor import ArmadaExecutor, InputPlacement

D = 8
WORKFLOW = "wf-visibility"
PROMPTS = ["galaxy alpha analysis", "galaxy beta analysis"]


class PairedBridge:
    """Two ranks in one process sharing one rendezvous slot list.

    allgather is a barrier: it deposits this rank's contribution in its own
    slot, waits for every participant, and only then returns all of them. A
    plain shared list is not enough — ranks driven one after another would let
    whichever ran first return with only its own table, which no executor
    ordering could repair. `barrier=None` is the single-rank case, where there
    is nothing to wait for.
    """

    world_size = 2
    available = True
    channel_type = "paired"

    def __init__(self, rank, rendezvous, barrier=None):
        self.rank = rank
        self._ctx = None
        self._rendezvous = rendezvous
        self._barrier = barrier

    def allgather(self, table):
        self._rendezvous[self.rank] = table
        if self._barrier is not None:
            self._barrier.wait()
        return [t for t in self._rendezvous if t is not None]


def _context_manager():
    from context.manager import ContextManager

    return ContextManager(backend="cylon", embedding_dim=D, persist_to_redis=False)


def _services(store):
    """Real ContextManager and ContextRouter; Bedrock mocked."""
    from context.router import ContextRouter

    embedding_service = MagicMock()
    embedding_service.embed.side_effect = lambda text: (
        np.full(D, float(len(text) % 5 + 1), dtype=np.float32),
        {"token_count": len(text)},
    )

    chain_executor = MagicMock()
    chain_executor.execute.side_effect = lambda prompt: {
        "response": f"analysis::{prompt[:20]}", "input_tokens": 3, "output_tokens": 5,
        "latency_ms": 1.0, "model_id": "mock-model",
    }

    router = ContextRouter(store, config=MagicMock(similarity_threshold=0.99), top_k=5)
    return embedding_service, router, store, chain_executor


def _run_rank(rank, store, rendezvous, barrier=None):
    services = _services(store)
    seq = build_cosmic_workflow(*services, workflow_id=WORKFLOW, dimensions=D)
    table = pa.table({"raw_text": [PROMPTS[rank]]}, schema=seq.operators[0].schema_in)
    ArmadaExecutor(PairedBridge(rank, rendezvous, barrier)).run(
        seq, input_tables=table, ctx=None, placement=InputPlacement.PreDistributed)
    return rendezvous


def test_a_context_published_on_one_rank_is_stored_on_the_other():
    """The minimal statement of all-to-all visibility: after the collective,
    each rank's own store holds both ranks' contexts."""
    rendezvous = [None, None]
    barrier = threading.Barrier(2, timeout=5)
    stores = [_context_manager(), _context_manager()]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_run_rank, rank, stores[rank], rendezvous, barrier)
                   for rank in (0, 1)]
        for future in futures:
            future.result()

    for rank, store in enumerate(stores):
        _, matrix = store.get_embedding_matrix(workflow_id=WORKFLOW)
        assert matrix.shape[0] == len(PROMPTS), (
            f"rank {rank} holds {matrix.shape[0]} contexts, expected both ranks' "
            f"{len(PROMPTS)} — the collective did not carry contexts across ranks"
        )


def test_the_collective_carries_contexts_not_acknowledgements():
    """Names the specific defect: an AllGather of a bool column moves no
    context, however many booleans it moves."""
    rendezvous = _run_rank(0, _context_manager(), [None, None])

    assert rendezvous, "nothing was handed to the collective at all"
    moved = rendezvous[0]
    assert moved.schema.field(0).name != "ack", (
        "the collective moved the ack column; acknowledgements are not contexts"
    )