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

"""What actually reaches Bedrock.

Every LLM mock in this suite echoes whatever prompt it is handed, so a pipeline
that sent Bedrock an empty string passed all of them. These tests assert the
prompt itself: on a miss it must be the galaxy's analysis text, and on a hit
there must be no call at all.

Run: pytest tests/armada/test_prompt_fidelity.py -v
"""

import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from armada.cosmic_workflow import build_cosmic_workflow

D = 8
WORKFLOW = "wf-prompt-fidelity"
PROMPT_A = "Galaxy A: predicted redshift 0.42 vs true 0.39, ugriz 18.2 17.5 17.0 16.7 16.5"
PROMPT_B = "Galaxy B: predicted redshift 0.88 vs true 0.85, ugriz 19.1 18.4 17.9 17.6 17.4"


def _store():
    from context.manager import ContextManager

    return ContextManager(backend="cylon", embedding_dim=D, persist_to_redis=False)


def _services(store, threshold, embed_value):
    """Real ContextManager and ContextRouter; Bedrock mocked and recording.

    embed_value decides hit versus miss: a constant makes every row identical
    so the second is a certain hit; a per-text value makes every row distinct
    so every row is a certain miss.
    """
    from context.router import ContextRouter

    seen = []

    embedding_service = MagicMock()
    embedding_service.embed.side_effect = lambda text: (
        embed_value(text), {"token_count": len(text)})

    chain_executor = MagicMock()

    def execute(prompt):
        seen.append(prompt)
        return {"response": f"ANALYSIS OF {prompt[:20]}", "input_tokens": 3,
                "output_tokens": 5, "latency_ms": 1.0, "model_id": "mock-model"}

    chain_executor.execute.side_effect = execute

    router = ContextRouter(store, config=MagicMock(similarity_threshold=threshold), top_k=5)
    return (embedding_service, router, store, chain_executor), seen


def _run(seq, prompts):
    table = pa.table({"raw_text": list(prompts)}, schema=seq.operators[0].schema_in)
    return seq.invoke(table)


def test_a_cache_miss_sends_the_galaxys_own_analysis_prompt():
    """The defect in one line: the prompt was the empty string, because
    Retrieve's ranked_docs carries a context id and Reason used it as the
    prompt. The analysis text rode through unused as raw_text."""
    distinct = lambda text: np.full(D, float(len(text) % 5 + 1), dtype=np.float32)
    services, seen = _services(_store(), threshold=0.999, embed_value=distinct)
    seq = build_cosmic_workflow(*services, workflow_id=WORKFLOW, dimensions=D)

    _run(seq, [PROMPT_A])

    assert seen == [PROMPT_A], (
        f"Bedrock received {seen!r}, not the galaxy's analysis prompt"
    )


def test_a_cache_hit_makes_no_llm_call_at_all():
    """H4's whole claim: a reused context avoids the call. While Reason called
    Bedrock unconditionally, llm_calls == galaxies at every reuse rate and the
    claim could not be measured."""
    same = lambda text: np.full(D, 1.0, dtype=np.float32)
    services, seen = _services(_store(), threshold=0.99, embed_value=same)
    seq = build_cosmic_workflow(*services, workflow_id=WORKFLOW, dimensions=D)

    _run(seq, [PROMPT_A])
    _run(seq, [PROMPT_B])

    assert seen == [PROMPT_A], (
        f"expected one call for the first galaxy only, got {len(seen)}: {seen!r}"
    )


def test_a_reused_context_is_not_stored_a_second_time():
    """A hit reuses an answer, so it must not deposit a copy of it. Storing a
    duplicate per hit would grow the store with near-identical vectors and slow
    every later similarity search — the search whose speed is the thesis."""
    same = lambda text: np.full(D, 1.0, dtype=np.float32)
    store = _store()
    services, seen = _services(store, threshold=0.99, embed_value=same)
    seq = build_cosmic_workflow(*services, workflow_id=WORKFLOW, dimensions=D)

    _run(seq, [PROMPT_A])
    _run(seq, [PROMPT_B])

    ids, matrix = store.get_embedding_matrix(workflow_id=WORKFLOW)
    assert matrix.shape[0] == 1, (
        f"{matrix.shape[0]} contexts stored for 2 galaxies, 1 of which was a hit"
    )
    stored = store.get_context(ids[0] if not hasattr(ids, "to_pylist") else ids.to_pylist()[0])
    assert stored["response"] == f"ANALYSIS OF {PROMPT_A[:20]}"
    assert seen == [PROMPT_A]