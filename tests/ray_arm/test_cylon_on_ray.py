"""Arm B placement and environment construction.

Ray is not required here: what matters is that the bundles match the world
size, that the placement strategy is validated rather than passed through, and
that the Cylon environment is built on the FMI transport. Running the arm needs
a cluster; checking that it is configured correctly does not.
"""

import pytest

from ray_arm.cylon_on_ray import (
    PLACEMENT_STRATEGIES, bundle_spec, resolve_strategy,
)


PEERS = "0:1,2;1:0,3"


def test_one_bundle_per_worker():
    assert bundle_spec(4) == [{"CPU": 1}] * 4


def test_bundles_honour_cpus_per_worker():
    assert bundle_spec(2, cpus_per_worker=4) == [{"CPU": 4}] * 2


def test_default_strategy_spreads_one_worker_per_node():
    """The headline comparison places one rank per node for every arm, so the
    default must match the other arms rather than Ray's own preference."""
    assert resolve_strategy(None) == "STRICT_SPREAD"


def test_pack_is_available_for_the_co_location_sensitivity_check():
    assert resolve_strategy("PACK") == "PACK"
    assert set(PLACEMENT_STRATEGIES) == {"STRICT_SPREAD", "PACK"}


def test_an_unknown_strategy_is_rejected_at_configuration_time():
    """A typo must not silently fall back to Ray's default, which would change
    placement without changing anything the results record."""
    with pytest.raises(ValueError) as excinfo:
        resolve_strategy("SPREAD_ISH")
    assert "SPREAD_ISH" in str(excinfo.value)


def test_actor_builds_a_cylon_env_on_the_fmi_transport(monkeypatch):
    """Arm B differs from Armada only in who schedules the workers. If this
    arm came up on Gloo, B to C would vary transport and scheduler at once and
    the decomposition the third arm exists to provide would be meaningless."""
    from ray_arm import cylon_on_ray

    built = {}

    class FakeBridge:
        def __init__(self, world_size, rank, channel_type, comm_name=None, **kwargs):
            self.rank = rank
            built["channel_type"] = channel_type
            built["rank"] = rank
            built["world_size"] = world_size

        @property
        def available(self):
            return True

        def finalize(self):
            built["finalized"] = True

    monkeypatch.setattr(cylon_on_ray, "FMIBridge", FakeBridge)

    actor = cylon_on_ray.CylonRayActor(rank=1, world_size=4,
                                       channel_type="redis", comm_name="run_a", required_peers=PEERS)
    assert actor.start_env() is True
    assert built["channel_type"] == "redis"
    assert built["rank"] == 1
    actor.shutdown()
    assert built["finalized"] is True


def test_actor_sources_from_the_environment_exactly_what_arm_c_sources(monkeypatch):
    """Arm B must reach the same coordinator Arm C does. If redis_host stayed
    at FMIBridge's empty-string default, a redis-channel run here could never
    find the Redis coordinator that Arm C uses, which is a second difference
    between the arms beyond the one this arm exists to isolate.

    The rendezvous pair is the mirror-image failure. Arm C never passes
    rendezvous_host, so it takes FMIBridge's default; rendezvous_host is
    threaded into FMIConfig on every channel type, so an arm that read
    RENDEZVOUS_HOST would diverge from Arm C wherever that variable is set.
    Both variables are set here to non-default values precisely so that
    reading either one would show up as a captured kwarg."""
    from ray_arm import cylon_on_ray

    built = {}

    class FakeBridge:
        def __init__(self, world_size, rank, channel_type, comm_name=None, **kwargs):
            self.rank = rank
            built.update(kwargs)

        @property
        def available(self):
            return True

        def finalize(self):
            pass

    monkeypatch.setattr(cylon_on_ray, "FMIBridge", FakeBridge)
    monkeypatch.setenv("RENDEZVOUS_HOST", "rendezvous.example")
    monkeypatch.setenv("RENDEZVOUS_PORT", "20000")
    monkeypatch.setenv("FMI_LISTEN_PORT", "30000")
    monkeypatch.setenv("REDIS_HOST", "redis.example")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("ADVERTISE_HOST", "advertise.example")
    monkeypatch.setenv("FMI_MAX_TIMEOUT", "300000")

    actor = cylon_on_ray.CylonRayActor(rank=0, world_size=2,
                                       channel_type="redis", comm_name="run_b", required_peers=PEERS)
    actor.start_env()

    assert "rendezvous_host" not in built
    assert "rendezvous_port" not in built
    assert built["listen_port"] == 30000
    assert built["redis_host"] == "redis.example"
    assert built["redis_port"] == 6380
    assert built["advertise_host"] == "advertise.example"
    assert built["maxtimeout"] == 300000


def test_channel_type_defaults_to_the_transport_arm_c_hardcodes(monkeypatch):
    """run_cosmic_local.py's Armada backend hardcodes channel_type to
    'direct-redis'. A caller that omits it here must land on the same
    transport, otherwise the arm that exists to isolate the scheduler would
    also be varying the channel whenever whoever launches these actors
    forgets the argument."""
    from ray_arm import cylon_on_ray

    built = {}

    class FakeBridge:
        def __init__(self, world_size, rank, channel_type, comm_name=None, **kwargs):
            self.rank = rank
            built["channel_type"] = channel_type

        @property
        def available(self):
            return True

        def finalize(self):
            pass

    monkeypatch.setattr(cylon_on_ray, "FMIBridge", FakeBridge)

    cylon_on_ray.CylonRayActor(rank=0, world_size=2,
                               comm_name="run_default", required_peers=PEERS).start_env()
    assert built["channel_type"] == "direct-redis"

    cylon_on_ray.CylonRayActor(rank=0, world_size=2, comm_name="run_override", required_peers=PEERS,
                               channel_type="redis").start_env()
    assert built["channel_type"] == "redis"


def test_maxtimeout_defaults_to_the_same_value_run_cosmic_local_uses(monkeypatch):
    """Arm C's Armada backend sources FMI_MAX_TIMEOUT with a default of 60000.
    If this arm fell through to FMIBridge's own constructor default of 120000
    instead, a collective that stalls under Ray's slower actor startup could
    time out on B while succeeding on C for a reason that has nothing to do
    with the scheduler."""
    from ray_arm import cylon_on_ray

    built = {}

    class FakeBridge:
        def __init__(self, world_size, rank, channel_type, comm_name=None, **kwargs):
            self.rank = rank
            built.update(kwargs)

        @property
        def available(self):
            return True

        def finalize(self):
            pass

    monkeypatch.setattr(cylon_on_ray, "FMIBridge", FakeBridge)
    monkeypatch.delenv("FMI_MAX_TIMEOUT", raising=False)

    actor = cylon_on_ray.CylonRayActor(rank=0, world_size=2,
                                       channel_type="redis", comm_name="run_c", required_peers=PEERS)
    actor.start_env()

    assert built["maxtimeout"] == 60000


def test_required_peers_passed_to_the_constructor_reaches_the_bridge(monkeypatch):
    """Arm C always computes a peer map before building its bridge. When the
    map is falsy FMIBridge pops FMI_REQUIRED_PEERS and the Direct channel
    eagerly connects every pair, N(N-1)/2 rendezvous pairings against Arm C's
    DAG edges only. establish_s is compared between the arms, so an omitted
    map would make this arm slower for a reason that is not the scheduler.
    The argument is therefore required, and this pins that it is forwarded."""
    from ray_arm import cylon_on_ray

    built = {}

    class FakeBridge:
        def __init__(self, world_size, rank, channel_type, comm_name=None, **kwargs):
            self.rank = rank
            built.update(kwargs)

        @property
        def available(self):
            return True

        def finalize(self):
            pass

    monkeypatch.setattr(cylon_on_ray, "FMIBridge", FakeBridge)

    peers = "0:1,2;1:0,3"
    actor = cylon_on_ray.CylonRayActor(rank=0, world_size=4,
                                       channel_type="direct-redis",
                                       comm_name="run_d", required_peers=peers)
    actor.start_env()

    assert built["required_peers"] == peers


def test_nonblocking_defaults_true_and_is_overridable(monkeypatch):
    """Arm C derives nonblocking from a command-line flag whose effective
    default is True; this arm must expose the same knob rather than hardcode
    a value the launcher cannot change."""
    from ray_arm import cylon_on_ray

    built = {}

    class FakeBridge:
        def __init__(self, world_size, rank, channel_type, comm_name=None, **kwargs):
            self.rank = rank
            built.update(kwargs)

        @property
        def available(self):
            return True

        def finalize(self):
            pass

    monkeypatch.setattr(cylon_on_ray, "FMIBridge", FakeBridge)

    default_actor = cylon_on_ray.CylonRayActor(rank=0, world_size=2,
                                               channel_type="redis", comm_name="run_e", required_peers=PEERS)
    default_actor.start_env()
    assert built["nonblocking"] is True

    overridden_actor = cylon_on_ray.CylonRayActor(rank=0, world_size=2,
                                                  channel_type="redis",
                                                  comm_name="run_f", required_peers=PEERS,
                                                  nonblocking=False)
    overridden_actor.start_env()
    assert built["nonblocking"] is False


def test_strategy_env_override_is_read_at_call_time(monkeypatch):
    """Configuration in this project resolves at call time, not import time,
    so RAY_PLACEMENT_STRATEGY must be honoured without reloading the module."""
    from ray_arm import cylon_on_ray

    monkeypatch.setenv("RAY_PLACEMENT_STRATEGY", "PACK")
    assert cylon_on_ray.resolve_strategy(None) == "PACK"


def test_the_actor_is_wired_to_the_real_fmi_bridge():
    """Every other test in this file monkeypatches FMIBridge, so all of them
    would still pass if this module imported a Gloo bridge under that name.
    That is the one property Arm B exists to guarantee: if the arm came up on
    Gloo, B against C would vary transport and scheduler at once and the
    decomposition the arm provides would be meaningless. Only an unmocked
    identity check can catch it."""
    import communicator.fmi_bridge as fmi_bridge
    from ray_arm import cylon_on_ray

    assert cylon_on_ray.FMIBridge is fmi_bridge.FMIBridge


def test_start_env_adopts_the_rank_fmi_assigns(monkeypatch):
    """FMI hands out ranks by Redis INCR during communicator construction, so
    the requested rank and the assigned rank routinely differ. Arm C rebuilds
    its whole workload on bridge.rank and stamps its record with it. If this
    arm kept the Ray-assigned rank it would compute one shard while
    participating in every collective as another: two ranks process the same
    galaxies, one shard is never processed, nothing raises, and the scaling
    curve is wrong in a direction nobody can see."""
    from ray_arm import cylon_on_ray

    class FakeBridge:
        def __init__(self, world_size, rank, channel_type, comm_name=None, **kwargs):
            self.rank = 3

        @property
        def available(self):
            return True

        def finalize(self):
            pass

    monkeypatch.setattr(cylon_on_ray, "FMIBridge", FakeBridge)

    actor = cylon_on_ray.CylonRayActor(rank=0, world_size=4,
                                       comm_name="run_rank", required_peers=PEERS)
    actor.start_env()

    assert actor.rank == 3
    assert actor.requested_rank == 0


def test_required_peers_cannot_be_omitted():
    """A None default would invite the omission that makes the Direct channel
    connect every pair. Nothing else in the arm forbids it, so the signature
    has to."""
    from ray_arm import cylon_on_ray

    with pytest.raises(TypeError):
        cylon_on_ray.CylonRayActor(rank=0, world_size=2, comm_name="run_missing")


def test_pack_placement_gives_each_rank_its_own_listen_port():
    """Under STRICT_SPREAD each rank owns a node and binds the base port,
    exactly as Arm C does. Under PACK ranks share a node; a fixed port would
    leave every rank after the first unable to bind, and FMIBridge swallows
    that failure, so the arm would run with no collectives and still write a
    result labelled ray-cylon."""
    from ray_arm import cylon_on_ray

    assert cylon_on_ray.resolve_listen_port(0, "PACK", base=10000) == 10000
    assert cylon_on_ray.resolve_listen_port(3, "PACK", base=10000) == 10003
    assert cylon_on_ray.resolve_listen_port(3, "STRICT_SPREAD", base=10000) == 10000


def test_actor_under_pack_binds_a_rank_specific_port(monkeypatch):
    """The offset has to reach the bridge, not merely exist as a helper."""
    from ray_arm import cylon_on_ray

    built = {}

    class FakeBridge:
        def __init__(self, world_size, rank, channel_type, comm_name=None, **kwargs):
            self.rank = rank
            built.update(kwargs)

        @property
        def available(self):
            return True

        def finalize(self):
            pass

    monkeypatch.setattr(cylon_on_ray, "FMIBridge", FakeBridge)
    monkeypatch.setenv("FMI_LISTEN_PORT", "10000")

    cylon_on_ray.CylonRayActor(rank=2, world_size=4, comm_name="run_pack",
                               required_peers=PEERS,
                               placement_strategy="PACK").start_env()
    assert built["listen_port"] == 10002

    cylon_on_ray.CylonRayActor(rank=2, world_size=4, comm_name="run_spread",
                               required_peers=PEERS,
                               placement_strategy="STRICT_SPREAD").start_env()
    assert built["listen_port"] == 10000


def test_bridge_is_exposed_after_start_env_and_cleared_on_shutdown(monkeypatch):
    from ray_arm import cylon_on_ray

    class FakeBridge:
        def __init__(self, world_size, rank, channel_type, comm_name=None, **kwargs):
            self.rank = rank

        @property
        def available(self):
            return True

        def finalize(self):
            pass

    monkeypatch.setattr(cylon_on_ray, "FMIBridge", FakeBridge)
    actor = cylon_on_ray.CylonRayActor(rank=0, world_size=2, comm_name="run_e", required_peers=PEERS)

    assert actor.bridge is None
    actor.start_env()
    assert isinstance(actor.bridge, FakeBridge)
    actor.shutdown()
    assert actor.bridge is None
