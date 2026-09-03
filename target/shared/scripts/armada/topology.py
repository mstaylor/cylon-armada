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

import math

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


def _rounds(world_size):
    return math.ceil(math.log2(world_size)) if world_size > 1 else 0


def binomial_tree_peers(world_size, rank, root=0):
    """Peers of `rank` in the binomial tree rooted at `root`."""
    peers = set()
    shifted = (rank - root) % world_size
    for i in range(_rounds(world_size)):
        step = 2 ** i
        partner = shifted + step
        if shifted % (2 * step) == 0 and partner < world_size:
            peers.add((partner + root) % world_size)
        elif shifted % step == 0 and shifted % (2 * step) != 0:
            peers.add((shifted - step + root) % world_size)
    peers.discard(rank)
    return peers


def recursive_doubling_peers(world_size, rank):
    """Peers of `rank` under recursive doubling (allreduce, barrier)."""
    peers = set()
    for i in range(_rounds(world_size)):
        partner = rank ^ (2 ** i)
        if partner < world_size:
            peers.add(partner)
    peers.discard(rank)
    return peers


def linear_gather_peers(world_size, rank, root=0):
    """Peers of `rank` for the linear gatherv payload phase (a star at root)."""
    if rank == root:
        return {p for p in range(world_size) if p != root}
    return {root}


def required_peers(world_size, rank, patterns, roots=(0,)):
    """Union of every peer `rank` can be asked to talk to under this plan.

    patterns: iterable of CollectivePattern actually present in the plan.
    roots:    iterable of roots those collectives run against.

    Returns an empty set for world_size <= 1 (nothing to connect).
    """
    if world_size <= 1:
        return set()

    patterns = set(patterns)
    peers = set()

    # A barrier is an allreduce, and the executor may issue one regardless of
    # which operator patterns the plan contains, so recursive-doubling peers are
    # always required — leaving them out risks a hang at the first barrier.
    peers |= recursive_doubling_peers(world_size, rank)

    for root in roots:
        if patterns & TREE_PATTERNS:
            peers |= binomial_tree_peers(world_size, rank, root)
        if patterns & GATHER_PATTERNS:
            peers |= linear_gather_peers(world_size, rank, root)

    peers.discard(rank)
    return peers


def required_peers_for_plan(plan, world_size, rank, roots=(0,)):
    """required_peers() driven straight off a compiled ExecutionPlan."""
    return required_peers(world_size, rank, set(plan.assignments.values()), roots)


def format_peer_list(peers):
    """Render a single rank's peer set."""
    return ",".join(str(p) for p in sorted(peers))


def peer_map(world_size, patterns, roots=(0,)):
    """required_peers() for every rank: {rank: {peers}}."""
    return {r: required_peers(world_size, r, patterns, roots) for r in range(world_size)}


def format_peer_map(peers_by_rank):
    """Render the whole map for FMI_REQUIRED_PEERS, as "0:1,2;1:0,3".

    Every rank is handed the full map rather than just its own row: FMI's Redis
    INCR counter assigns the real rank *after* the communicator is constructed,
    so a process cannot know which row is its own at the time this value has to
    be set. Direct::connection_targets() selects the row once peer_id is final.
    Handing over a pre-selected row would silently apply another rank's peers and
    break pairing symmetry — one side waits for a connection the other never makes.
    """
    return ";".join(
        f"{rank}:{format_peer_list(peers)}"
        for rank, peers in sorted(peers_by_rank.items())
    )