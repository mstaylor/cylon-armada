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

"""Epoch/batch planning for the Experiment E comparison.

Reuse is only observable across epochs: within one pass Retrieve queries a
cache nothing has populated yet. Batch size therefore sets how much
data-plane activity a run contains and how many generations the cache grows
through, which makes it a parameter of the experiment rather than an
implementation detail.

Run: pytest tests/armada/test_epochs.py -v
"""

import os
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTS = os.path.join(_REPO, "target", "shared", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from armada.epochs import epoch_count, plan_epochs


@pytest.mark.parametrize("n_local,batch_size,expected", [
    (19, 4, [(0, 4), (4, 8), (8, 12), (12, 16), (16, 19)]),
    (8, 4, [(0, 4), (4, 8)]),
    (3, 4, [(0, 3)]),
    (1, 1, [(0, 1)]),
])
def test_batches_tile_the_shard_exactly(n_local, batch_size, expected):
    assert plan_epochs(n_local, batch_size) == expected


def test_every_row_appears_exactly_once():
    covered = [i for start, stop in plan_epochs(19, 4) for i in range(start, stop)]
    assert covered == list(range(19))


def test_weak_scaling_load_gives_five_epochs():
    """19 galaxies per rank at the default batch size of 4 — five sharing steps
    per run, and a cache with five generations to grow through."""
    assert epoch_count(19, 4) == 5


def test_a_rank_with_no_galaxies_has_no_epochs():
    """At world sizes above the population some ranks get an empty shard; they
    must still not enter a batch."""
    assert plan_epochs(0, 4) == []
    assert epoch_count(0, 4) == 0


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        plan_epochs(10, 0)