"""Arm A: context shared through Ray's object store.

Each rank owns a shard and publishes its new contexts with ray.put, which
places them in Plasma; the registry actor holds only the references, so a
reader fetches the payload directly from the object store rather than through
the registry. That keeps the registry small and lets Ray decide whether a read
is a shared-memory mapping or a network transfer.

No barrier is imposed. Ray may resolve references lazily and overlap them,
which is the behaviour a Ray deployment would actually exhibit.

Payloads are expected to be numpy arrays or Arrow buffers: those are the
objects Plasma places and returns zero-copy, which is the property this arm
exists to compare against. Plain Python objects still work through
``ray.put``/``ray.get``, but fall back to pickling and do not exercise that
property.
"""

import ray


@ray.remote
class ContextRegistry:
    """Directory of object references, in the order ranks published them.

    Callers should read at batch granularity through the ``since`` watermark
    rather than once per item. A per item read pattern would turn this single
    actor into a synchronisation point once a run reaches dozens of workers
    (for example 64).
    """

    def __init__(self):
        self._log = []

    def register(self, rank, refs):
        self._log.extend((rank, ref) for ref in refs)

    def all_refs(self, since=0):
        new_refs = [ref for _, ref in self._log[since:]]
        return new_refs, len(self._log)

    def count(self):
        return len(self._log)


@ray.remote
class ShardActor:
    """One rank's worth of work, sharing context through the object store."""

    def __init__(self, rank, world_size, registry):
        self.rank = rank
        self.world_size = world_size
        self._registry = registry
        self._contexts = []
        self._watermark = 0
        self._pending_registration = None

    def publish(self, objects):
        refs = [ray.put(obj) for obj in objects]
        self._pending_registration = self._registry.register.remote(self.rank, refs)
        return len(refs)

    def flush(self):
        """Wait for this rank's most recent publish to land in the registry.

        Not part of the read/write hot path: publish stays fire-and-forget so
        no rank ever blocks on another. This exists so a caller that needs to
        observe a specific write, such as a test, can wait for it explicitly
        instead of the write imposing that wait on everyone.
        """
        if self._pending_registration is not None:
            ray.get(self._pending_registration)

    def watermark(self):
        """How far into the registry log this rank has already read.

        Exposed so the incremental-fetch property can be asserted against this
        read path rather than against the registry alone.
        """
        return self._watermark

    def visible_contexts(self):
        """Every context this rank can see, fetching only what is new.

        The fetch is incremental: the watermark means a read costs one
        ray.get over the references added since the last read, and nothing
        when nothing was published. The returned value is the full visible
        set, so it grows with the run; callers on a hot path should hold the
        result rather than calling this per item.
        """
        new_refs, total = ray.get(self._registry.all_refs.remote(since=self._watermark))
        if new_refs:
            self._contexts.extend(ray.get(new_refs))
        self._watermark = total
        return list(self._contexts)
