from concurrent.futures import ThreadPoolExecutor

import pytest

ray = pytest.importorskip("ray")

from ray_arm.exchange_board import ExchangeBoard


@pytest.fixture(scope="module")
def local_ray():
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, namespace="board-test")
    yield
    ray.shutdown()


def test_collect_returns_every_ranks_ref_in_rank_order(local_ray):
    board = ExchangeBoard.remote()
    refs = {rank: ray.put(f"from-{rank}") for rank in (2, 0, 1)}
    for rank, ref in refs.items():
        board.post.remote(0, rank, [ref])

    ref_lists = ray.get(board.collect.remote(0, 3, 1), timeout=30)

    assert ray.get([ref_list[0] for ref_list in ref_lists]) == ["from-0", "from-1", "from-2"]


def test_collect_waits_until_the_expected_posts_arrive(local_ray):
    board = ExchangeBoard.remote()
    board.post.remote(0, 0, [ray.put("a")])
    pending = board.collect.remote(0, 2, 1)

    ready, _ = ray.wait([pending], timeout=0.5)
    assert not ready

    board.post.remote(0, 1, [ray.put("b")])
    assert len(ray.get(pending, timeout=30)) == 2


def test_a_rank_rounds_ahead_does_not_leak_into_the_current_round(local_ray):
    board = ExchangeBoard.remote()
    for round_number in (0, 1, 2):
        board.post.remote(round_number, 0, [ray.put(f"r{round_number}-0")])
    board.post.remote(0, 1, [ray.put("r0-1")])

    ref_lists = ray.get(board.collect.remote(0, 2, 2), timeout=30)

    assert ray.get([ref_list[0] for ref_list in ref_lists]) == ["r0-0", "r0-1"]


def test_a_round_is_dropped_only_after_every_reader_collects_it(local_ray):
    board = ExchangeBoard.remote()
    board.post.remote(0, 0, [ray.put("x")])
    board.post.remote(1, 0, [ray.put("y")])

    ray.get(board.collect.remote(0, 1, 2), timeout=30)
    assert ray.get(board.rounds_held.remote()) == [0, 1]

    ray.get(board.collect.remote(0, 1, 2), timeout=30)
    assert ray.get(board.rounds_held.remote()) == [1]


def test_concurrent_collectors_all_receive_the_round(local_ray):
    board = ExchangeBoard.remote()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(ray.get, board.collect.remote(0, 2, 3), timeout=30) for _ in range(3)]
        board.post.remote(0, 0, [ray.put("a")])
        board.post.remote(0, 1, [ray.put("b")])
        assert all(len(f.result(timeout=30)) == 2 for f in futures)
    assert ray.get(board.rounds_held.remote()) == []