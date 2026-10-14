import math
import os
import subprocess
import sys

import pytest

from armada.peer_sets import (
    GatherTopology,
    binomial_tree_peers,
    collective_peer_map,
    linear_gather_peers,
    preconnected_gather_peers,
    recursive_doubling_peers,
)

WORLD_SIZES = [1, 2, 3, 5, 6, 13, 517]


@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("root", [0, "last"])
def test_linear_gather_preconnects_the_star_and_binomial_preconnects_nothing(world_size, root):
    root = world_size - 1 if root == "last" else root
    for rank in range(world_size):
        assert preconnected_gather_peers(world_size, rank, root, GatherTopology.LINEAR) == \
            linear_gather_peers(world_size, rank, root)
        assert preconnected_gather_peers(world_size, rank, root, "binomial") == set()


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_binomial_gather_without_barrier_or_tree_preconnects_no_peer(world_size):
    peers = collective_peer_map(world_size, tree=False, gather=True,
                                gather_topology=GatherTopology.BINOMIAL, barrier=False)
    assert peers == {r: set() for r in range(world_size)}


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_binomial_gather_with_tree_collectives_preconnects_the_tree_without_the_star(world_size):
    peers = collective_peer_map(world_size, tree=True, gather=True,
                                gather_topology=GatherTopology.BINOMIAL, barrier=False)
    assert peers == {r: (binomial_tree_peers(world_size, r, 0) if world_size > 1 else set())
                     for r in range(world_size)}
    if world_size > 1:
        assert len(peers[0]) == math.ceil(math.log2(world_size))


@pytest.mark.parametrize("world_size", [2, 5, 13, 517])
def test_linear_topology_reproduces_the_default_star(world_size):
    assert collective_peer_map(world_size, tree=True, gather=True, gather_topology=GatherTopology.LINEAR) == \
        collective_peer_map(world_size, tree=True, gather=True)


def test_barrier_adds_recursive_doubling_peers_to_any_topology():
    with_barrier = collective_peer_map(8, tree=True, gather=True, gather_topology="binomial", barrier=True)
    without = collective_peer_map(8, tree=True, gather=True, gather_topology="binomial", barrier=False)
    assert without[3] == {2}
    assert with_barrier[3] == {2} | recursive_doubling_peers(8, 3)


def test_unknown_gather_topology_is_rejected():
    with pytest.raises(ValueError):
        preconnected_gather_peers(4, 1, 0, "star")


topology =pytest.importorskip("armada.topology")
CollectivePattern = topology.CollectivePattern

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..", "target", "shared", "scripts")


@pytest.mark.parametrize("world_size", [1, 2, 5, 11, 41, 164])
@pytest.mark.parametrize("patterns,tree,gather", [
    ((CollectivePattern.Broadcast, CollectivePattern.ScatterGather), True, True),
    ((CollectivePattern.Broadcast,), True, False),
    ((), False, False),
])
def test_collective_peer_map_matches_the_pattern_driven_map(world_size, patterns, tree, gather):
    for root in (0, world_size // 2):
        assert collective_peer_map(world_size, tree=tree, gather=gather, roots=(root,)) == \
            topology.peer_map(world_size, patterns, roots=(root,))


def test_importing_peer_sets_loads_no_native_cylon_module():
    probe = "import sys, armada.peer_sets; print(any(m.startswith(('cylon_armada', 'pycylon')) for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", probe], env={**os.environ, "PYTHONPATH": os.path.abspath(_SCRIPTS)},
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
