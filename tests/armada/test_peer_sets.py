import os
import subprocess
import sys

import pytest

from armada.peer_sets import collective_peer_map

topology = pytest.importorskip("armada.topology")
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
