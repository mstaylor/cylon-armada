"""The rendezvous ray-objectstore's Experiment B collectives exchange refs through.

Each collective call is one numbered round. Ranks post the refs they put for
that round and collect the round's refs once the expected number of posts has
arrived. Rounds are kept apart by number because warmup reps run without a
barrier, so a rank that only sends (a broadcast root, a gather contributor)
can run rounds ahead of the ranks that read. A round is deleted once all of
its readers have collected it.

Refs are posted inside a list so Ray passes the ref itself rather than
resolving it to the value.
"""

import asyncio

import ray


@ray.remote
class ExchangeBoard:
    """Per-round ref exchange. Async, so a collecting rank waits on a condition
    inside the actor without blocking later posts."""

    def __init__(self):
        self._posts = {}
        self._reads_left = {}
        self._condition = asyncio.Condition()

    async def post(self, round_number, rank, ref_list):
        async with self._condition:
            self._posts.setdefault(round_number, {})[rank] = ref_list
            self._condition.notify_all()

    async def collect(self, round_number, expected, readers):
        """The round's posted ref lists in rank order, once `expected` have
        arrived. The round is deleted after `readers` collects."""
        async with self._condition:
            await self._condition.wait_for(
                lambda: len(self._posts.get(round_number, {})) >= expected)
            posted = self._posts[round_number]
            ref_lists = [posted[rank] for rank in sorted(posted)]
            left = self._reads_left.get(round_number, readers) - 1
            if left <= 0:
                self._posts.pop(round_number, None)
                self._reads_left.pop(round_number, None)
            else:
                self._reads_left[round_number] = left
            return ref_lists

    async def rounds_held(self):
        async with self._condition:
            return sorted(self._posts)