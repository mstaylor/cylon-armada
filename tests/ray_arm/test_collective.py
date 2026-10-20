"""ray-collective's exchange through ray.util.collective, on a local Ray.

The group is real: two actors joined by torch.distributed Gloo, because the
property under test is that Ray's own collective delivers every rank's bytes
to every rank, which a mock could only assume.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

ray = pytest.importorskip("ray")
np = pytest.importorskip("numpy")
pa = pytest.importorskip("pyarrow")

from ray_arm.collective import (GlooInterfaceNotFound, exchanger_options, form_collective_group,
                                pack_table, resolve_gloo_interface, unpack_buffer)

KV = pa.struct([pa.field("k", pa.large_utf8()), pa.field("v", pa.large_utf8())])


@pytest.fixture(scope="module")
def local_ray():
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, namespace="coll-test")
    yield
    ray.shutdown()


def _rows(n, tag):
    return pa.table({"kv_pairs": pa.array([{"k": f"{tag}{i}", "v": "x" * (10 + i)} for i in range(n)],
                                          type=KV)})


def test_a_packed_table_unpacks_to_the_same_table():
    table = _rows(3, "a")
    buffer = pack_table(table)

    assert buffer.dtype == np.uint8
    assert unpack_buffer(buffer).equals(table)


def test_an_empty_table_still_packs_its_schema():
    """A rank with nothing new must still join every epoch's collective, so its
    empty contribution has to survive the round trip as an empty table."""
    empty = _rows(0, "e")
    restored = unpack_buffer(pack_table(empty))

    assert restored.num_rows == 0
    assert restored.schema == empty.schema


def test_the_gloo_interface_is_passed_to_the_actor_only_when_configured(monkeypatch):
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    assert "runtime_env" not in exchanger_options(0, "run", "node", "ns")

    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "eth1")
    options = exchanger_options(0, "run", "node", "ns")
    assert options["runtime_env"] == {"env_vars": {"GLOO_SOCKET_IFNAME": "eth1"}}
    assert options["name"] == "cosmic_exchanger_run_0"
    assert options["namespace"] == "ns"


FARGATE_LIKE_INTERFACES = {"lo": ["127.0.0.1"], "eth0": ["169.254.172.2"],
                           "eth1": ["10.0.1.23"], "docker0": ["172.17.0.1"]}


def test_the_interface_is_the_one_holding_the_node_ip():
    """Gloo must bind the address Ray itself advertises for the node, not
    whatever the container hostname resolves to."""
    assert resolve_gloo_interface(0, "10.0.1.23", FARGATE_LIKE_INTERFACES) == "eth1"


def test_an_explicitly_configured_interface_wins():
    assert resolve_gloo_interface(0, "10.0.1.23", FARGATE_LIKE_INTERFACES,
                                  configured="ens5") == "ens5"


def test_an_explicit_interface_wins_even_when_no_interface_holds_the_node_ip():
    assert resolve_gloo_interface(2, "10.9.9.9", FARGATE_LIKE_INTERFACES,
                                  configured="ens5") == "ens5"


def test_no_interface_holding_the_node_ip_is_an_error_naming_rank_ip_and_interfaces():
    with pytest.raises(GlooInterfaceNotFound) as caught:
        resolve_gloo_interface(3, "10.0.1.99", FARGATE_LIKE_INTERFACES)
    message = str(caught.value)
    assert "rank 3" in message and "10.0.1.99" in message
    assert all(name in message for name in FARGATE_LIKE_INTERFACES)


def test_a_loopback_node_ip_is_an_error_rather_than_a_silent_local_bind():
    """A node advertised on loopback is unreachable from every other task, so
    binding Gloo there would hang the group instead of failing."""
    with pytest.raises(GlooInterfaceNotFound, match="rank 0.*127.0.0.1"):
        resolve_gloo_interface(0, "127.0.0.1", {"lo": ["127.0.0.1"]})


def test_the_exchanger_binds_gloo_to_the_interface_holding_its_node_ip(local_ray, monkeypatch):
    """Resolved inside the actor, since on Fargate it runs on its own task."""
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    exchanger = form_collective_group(0, 1, "coll-iface", "coll-test", ray,
                                      timeout_s=30.0, poll_s=0.05)

    chosen = ray.get(exchanger.gloo_interface.remote(), timeout=60)

    assert chosen["source"] == "derived"
    assert chosen["node_ip"] == ray.util.get_node_ip_address()
    import psutil
    held = [a.address for a in psutil.net_if_addrs()[chosen["interface"]]]
    assert chosen["node_ip"] in held


def test_every_rank_receives_every_ranks_bytes_in_rank_order(local_ray):
    """Contributions differ in length, so the exchange must carry each rank's
    true size rather than the padded width."""
    group = "coll-order"
    rank1 = form_collective_group(1, 2, group, "coll-test", ray, timeout_s=30.0, poll_s=0.05)
    rank0 = form_collective_group(0, 2, group, "coll-test", ray, timeout_s=30.0, poll_s=0.05)
    payloads = [pack_table(_rows(2, "zero")), pack_table(_rows(5, "one"))]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ray.get, actor.allgather.remote(payload))
                   for actor, payload in ((rank0, payloads[0]), (rank1, payloads[1]))]
        results = [future.result(timeout=60) for future in futures]

    for gathered, collective_s in results:
        assert [unpack_buffer(b).num_rows for b in gathered] == [2, 5]
        assert unpack_buffer(gathered[1]).equals(unpack_buffer(payloads[1]))
        assert collective_s >= 0.0


def test_a_single_rank_group_returns_its_own_contribution(local_ray):
    solo = form_collective_group(0, 1, "coll-solo", "coll-test", ray, timeout_s=30.0, poll_s=0.05)
    payload = pack_table(_rows(1, "s"))

    gathered, _ = ray.get(solo.allgather.remote(payload), timeout=60)

    assert len(gathered) == 1
    assert unpack_buffer(gathered[0]).equals(unpack_buffer(payload))


def test_rank_zero_refuses_to_wait_forever_for_a_peer_that_never_appears(local_ray):
    with pytest.raises(TimeoutError, match="coll-missing"):
        form_collective_group(0, 2, "coll-missing", "coll-test", ray, timeout_s=0.3, poll_s=0.05)
