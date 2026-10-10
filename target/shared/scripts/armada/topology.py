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

"""Which peers a rank actually talks to, derived from a compiled ExecutionPlan.

FMI's direct channel eagerly establishes a connection to every other rank before
running anything, because the channel has no idea which peers a non-blocking
collective will need and a missing pairing at collective time is unrecoverable.
That costs N(N-1)/2 rendezvous pairings — 14,878 at the world_size=173 the
original Cosmic AI Lambda runs used.

cylon-armada does know the schedule ahead of time: compile_workflow() names every
operator, its CollectivePattern, and its root before a byte moves. So the exact
peer set is derivable up front, which keeps the "established before needed"
guarantee while connecting only the edges the plan will use. Feeding the result to
FMIBridge(required_peers=...) sets FMI_REQUIRED_PEERS, which Direct::init() honors.

Peer sets mirror the algorithms in cylon's PeerToPeer.cpp:
  - binomial tree (scatter/scatterv/gather-sizes/bcast/reduce): rank +/- 2^i
  - recursive doubling (allreduce, and therefore barrier): rank XOR 2^i
  - linear gatherv (the variable-length payload half of a Table gather): every
    non-root sends straight to root, so root's degree stays N-1 for those ops

An empty result means "no restriction" and leaves the full mesh in place.
"""

from armada.peer_sets import (
    binomial_tree_peers,
    collective_peers,
    format_peer_list,
    format_peer_map,
    linear_gather_peers,
    recursive_doubling_peers,
)
from cylon_armada.dag_compiler import CollectivePattern

TREE_PATTERNS = frozenset({
    CollectivePattern.Scatter,
    CollectivePattern.ScatterGather,
    CollectivePattern.Reduce,
    CollectivePattern.Broadcast,
})

GATHER_PATTERNS = frozenset({
    CollectivePattern.ScatterGather,
    CollectivePattern.Reduce,
})


def required_peers(world_size, rank, patterns, roots=(0,)):
    """Union of every peer `rank` can be asked to talk to under this plan.

    patterns: iterable of CollectivePattern actually present in the plan.
    roots:    iterable of roots those collectives run against.

    Returns an empty set for world_size <= 1 (nothing to connect).
    """
    patterns = set(patterns)
    return collective_peers(world_size, rank, tree=bool(patterns & TREE_PATTERNS),
                            gather=bool(patterns & GATHER_PATTERNS), roots=roots)


def required_peers_for_plan(plan, world_size, rank, roots=(0,)):
    """required_peers() driven straight off a compiled ExecutionPlan."""
    return required_peers(world_size, rank, set(plan.assignments.values()), roots)


def peer_map(world_size, patterns, roots=(0,)):
    """required_peers() for every rank: {rank: {peers}}."""
    return {r: required_peers(world_size, r, patterns, roots) for r in range(world_size)}
