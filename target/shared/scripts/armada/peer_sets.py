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


"""Peer sets of FMI's point-to-point collective algorithms, as plain rank arithmetic.

Kept free of native imports so it can run where Cylon's native libraries must not
load before mpi4py is configured, such as the Cosmic AI Lambda executor.
armada.topology maps CollectivePatterns onto these sets.
"""

import math


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


def collective_peers(world_size, rank, tree, gather, roots=(0,)):
    """Every peer `rank` can be asked to talk to by a barrier, plus binomial-tree
    collectives when `tree` and the linear gatherv star when `gather`, for each root.

    Returns an empty set for world_size <= 1 (nothing to connect).
    """
    if world_size <= 1:
        return set()

    # A barrier is an allreduce, and the executor may issue one regardless of
    # which operator patterns the plan contains, so recursive-doubling peers are
    # always required — leaving them out risks a hang at the first barrier.
    peers = recursive_doubling_peers(world_size, rank)

    for root in roots:
        if tree:
            peers |= binomial_tree_peers(world_size, rank, root)
        if gather:
            peers |= linear_gather_peers(world_size, rank, root)

    peers.discard(rank)
    return peers


def collective_peer_map(world_size, tree, gather, roots=(0,)):
    """collective_peers() for every rank: {rank: {peers}}."""
    return {r: collective_peers(world_size, r, tree, gather, roots) for r in range(world_size)}


def format_peer_list(peers):
    """Render a single rank's peer set."""
    return ",".join(str(p) for p in sorted(peers))


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
