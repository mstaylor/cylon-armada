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

"""Epoch planning for the Cosmic AI comparison runs.

A rank processes its shard in batches, sharing new contexts after each one, so
the semantic cache accumulates across epochs and the reuse rate becomes
observable. One batch covering the whole shard reduces to the degenerate case
where nothing is ever reused: Retrieve would query a cache nothing has
populated yet, and the sharing step's payload would never be read.

Batch size is therefore a parameter of the experiment rather than an
implementation detail — it sets both how many sharing steps a run contains
and how many generations the cache grows through — and is recorded with every
result.
"""


def plan_epochs(n_local, batch_size):
    """Contiguous [start, stop) batches tiling range(n_local)."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [(start, min(start + batch_size, n_local))
            for start in range(0, n_local, batch_size)]


def epoch_count(n_local, batch_size):
    """Number of epochs a rank runs for this shard size."""
    return len(plan_epochs(n_local, batch_size))