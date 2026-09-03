# Research Progress — 27 Aug to 3 Sep 2026

**Toward an agentic Cosmic AI pipeline executable on AWS Lambda**

Scope of this note: progress against the proposal's hypotheses H1–H5 and the Experiment A–L map, and
answers to three questions raised on the Experiment B benchmark figures.

---

## 1. Where this sits in the proposal

The thesis claim is that **data orchestration, not LLM inference, is the bottleneck in distributed
agentic pipelines**, and that compiling a fixed-topology agentic DAG into a deterministic
collective-communication schedule over zero-copy Arrow removes that bottleneck without touching
inference. The work of the past week advanced four items on the experiment map:

| Exp | Hyp. | Status on 27 Aug | Status on 3 Sep |
|---|---|---|---|
| **B** — collective latency/throughput, N=4–64 | H1 | ◑ partial, Lambda data only | ✅ complete cross-platform dataset: Rivanna (UCC) vs Fargate (FMI direct-redis), N=1–64, 8 collectives × 7 payload sizes |
| **A2** — Arrow schema-compatibility validation | H2 | ⏳ pending (synthetic cases only) | ◑ enforcement now exercised on a real agentic workflow — and it **rejected the proposal's own operator chain** (§2.1) |
| **E** — end-to-end pipeline speedup S(N) | — | ◑ HTTP arm only | ◑ executor + operator algebra built and running locally at N ∈ {1,2,3,4,5,8}; swap-equivalence property established (§2.3) |
| **I, L** (and the Lambda arm of E) | — | blocked | unblocked: a structural scaling limit in connection establishment was identified and removed (§2.4) |

The four findings below are stated as research results rather than implementation notes, because each
one changes something the proposal asserts or assumes.

---

## 2. Findings

### 2.1 The five-operator algebra is not closed under composition

The proposal defines five typed operators `O = (pattern, schema_in, schema_out)` and states that an
edge is legal — and therefore zero-copy eligible — exactly when `schema_out(Oᵢ) == schema_in(Oⱼ)`.
Composing the canonical five into the Cosmic AI pipeline and running them through the compiler
produces a **hard rejection at the last edge**:

```
SchemaMismatchError on edge Reason -> MemoryUpsert:
  producer  large_string
  consumer  struct<k: large_string, v: large_string>
```

This is not an implementation defect. `Reason` emits response text; `MemoryUpsert` consumes a keyed
`(context_id, envelope)` pair; nothing in the five operators constructs that key. **The operator set
as published is not closed under composition** — a reviewer composing the five as specified will hit
this immediately.

Two ways to close it: admit an adapter/binding operator class into the algebra, or widen
`MemoryUpsert`'s input contract. I implemented the first (a `Bind` operator, `PointToPoint`, no
collective) because it leaves the five validated A2 contracts untouched and makes the seam explicit
rather than hiding a conversion inside `MemoryUpsert`. Either way **the proposal should state which**,
since the composability claim is load-bearing for H2.

Worth noting the A2 machinery behaved exactly as designed here — it caught a real type error in the
proposal's own specification, at compile time, before any data moved. That is the strongest evidence
to date that A2 enforcement is doing useful work rather than decorating the pipeline.

### 2.2 The canonical contracts are lossy for the semantic-reuse mechanism (H4)

Each canonical operator declares exactly one column. Following the data through the chain, the
`task_description` and the embedding computed at `Embed` are both destroyed before reaching
`MemoryUpsert`, because neither appears in the intervening contracts (`ranked_docs`, then `response`).

The consequence is not cosmetic. A context stored without its originating embedding cannot be found
by `ContextRouter.find_similar` — cosine similarity against a placeholder vector matches nothing — so
**the semantic hit-rate `h` that H4 depends on would be structurally zero for anything written
through the specified chain**, while every individual operator still satisfied its declared contract.

Resolved by carrying provenance (`raw_text`, `query_embedding`, LLM token usage) as additive
passthrough columns alongside the canonical one, so the declared A2 contracts are unchanged and the
compile-time check still governs the edge. The finding for the proposal is that **the operator
contracts and the reuse mechanism are in tension, and the document should say how provenance
travels** — otherwise H4's `h ≥ 0.70` target is unreachable by construction.

### 2.3 Swap-equivalence established for Experiment E

The same `ArmadaSequence` now executes two ways — through LangChain's native runtime, and through the
compiled collective schedule — and at `world_size == 1` the two produce identical results by
construction. Establishing this required correcting the executor's dispatch order: `Scatter` must
move data *before* the operator body runs (each rank needs its shard before it can compute), whereas
every other pattern computes first and then moves its contribution.

This matters beyond correctness: it is the control condition for E. A speedup `S(N)` is only
attributable to the data plane if the *same* authored pipeline is what ran in both arms, and that is
now demonstrable rather than asserted.

### 2.4 The compiled schedule yields a connection topology a dynamic runtime cannot derive

This is the most consequential result of the week and, I think, a contribution the proposal should
claim explicitly.

FMI's direct channel eagerly establishes a connection to **every** peer before running anything —
N(N−1)/2 pairings — and it must, because a general-purpose runtime cannot know which peers a
non-blocking collective will need, and a pairing that is missing at collective time is unrecoverable.
At the world sizes the original Cosmic AI Lambda deployment actually used (**N = 110, 137, 173**, from
the captured invocation logs) that is 5,995 to 14,878 rendezvous pairings, walked serially, against a
60-second registration expiry on the rendezvous server.

Because cylon-armada compiles the DAG *before* execution, the exact peer set is known in advance —
the union of binomial-tree neighbours, recursive-doubling partners, and the gather star. Connections
drop from N−1 per rank to ~log N for every non-root rank. Measured at N = 16 on a Table-gather
workload:

| establishment strategy | ranks completed | max establish time |
|---|---|---|
| serial, full mesh (prior behaviour) | **9 of 16** — did not complete in 10 min | **289 s** |
| parallel, full mesh | 16 | 0.65 s |
| parallel, plan-derived topology | 16 | **0.33 s** |

Serial full-mesh establishment at N = 16 did not merely scale badly, it **failed to complete** — which
means the serverless arm was capped well below the world sizes the science workload actually runs at.

The research argument: this is a *second, independent* benefit of compiling the agentic DAG ahead of
time, orthogonal to zero-copy. The schedule is what makes the communication topology knowable, and the
topology is what makes the serverless arm feasible at realistic N. Zero-copy is an argument about
bytes on an edge; this is an argument about which edges need to exist at all. Both follow from the
same compile-first premise.

An accompanying correctness result: anything derived from a rank must be derived *after* rank
assignment. The FMI communicator reassigns ranks through a Redis `INCR` counter, so a topology
computed from the requested rank silently applies one rank's peer set to a different rank, breaking
pairing symmetry. It presented as a hang at N = 4 and 8 while 2, 3 and 5 passed.

### 2.5 Measurement-validity corrections to Experiment B

Two defects would have invalidated the H1 numbers had they gone unnoticed:

1. **A missing per-round progress drain** in non-blocking `allgather`/`allgatherv`/`bcast`. Rounds of
   the binomial tree could overlap, producing wrong or hung results rather than an obvious failure.
2. **Non-uniform vCPU allocation** across the sweep. Task size determines network tier on Fargate, so
   mixing 2- and 4-vCPU tasks made the platform comparison unfair.

Both corrected, and the **entire sweep re-run uniformly at 2 vCPU** (chosen so that N = 64 fits inside
the 140-vCPU account quota). The figures discussed below are from that re-run.

Separately, collective correctness was verified at **arbitrary, non-power-of-two world sizes** (N = 3,
5) with real distributed collectives. This matters because the original Cosmic AI runs used N = 110,
137, 173 — arbitrary N is the operating regime for the serverless arm, not a corner case.

---

## 3. Answers to the questions on the benchmark figures

Figure mapping assumed below (please correct if it differs): **Fig. 1** = barrier latency vs N,
**Fig. 2** = collective latency vs N (all collectives on one axes), **Fig. 3** = throughput vs message
size at N = 64.

### 3.1 Can Figure 2 be decomposed into per-collective subplots?

Yes, and it should be. A 2×4 grid — scatter / scatterv, gather / allgather, reduce / allreduce,
broadcast / barrier — with a shared log-log frame and Rivanna vs Fargate as the two series in each
panel.

Beyond legibility, decomposition is what makes the anomalies in 3.2 and 3.3 *readable*. Both are
payload-dependent effects that appear in some collectives and not others; overlaying eight collectives
on one axes averages that structure away and leaves only an unexplained spread. The existing chart
conventions carry over unchanged (box frame, no gridlines, major ticks only on log axes, error bars
with `capsize=5`, legend in a bordered box below the plot).

### 3.2 Why does gather on Fargate show a scaling anomaly?

**It is payload-driven, not rank-driven** — which the current figure obscures. At small payloads
Fargate gather scales smoothly across the whole range (8 B–32 KB: 0.005 ms at N=1 to 3.7 ms at N=64).
The departure is confined to the two largest payloads (median latency, ms):

| payload | N=8 | N=16 | N=32 | N=64 |
|---|---|---|---|---|
| 32 KB | 1.60 | 2.11 | 2.52 | 3.73 |
| 256 KB | 2.06 | 3.65 | **16.27** | **241.77** |
| 1 MB | 7.64 | 17.65 | **210.53** | **284.17** |

**Mechanism.** A Table gather transmits its variable-length payload through a *linear* gatherv — every
non-root rank sends directly to root, by design, matching UCC's `tl/ucp gatherv_linear`. At N=64 with
1 MB per rank, 64 MB converges on a single task's NIC from 63 simultaneous senders.

**Evidence that it is incast collapse rather than plain bandwidth saturation.** At N = 64 the two
large payloads achieve very different goodput:

- 256 KB: 16.8 MB in 241.8 ms → **69 MB/s**
- 1 MB: 67.1 MB in 284.2 ms → **236 MB/s**

The *smaller* transfer sustains 3.4× *lower* goodput, and it is reproducible (σ = 5.9 ms, CV 2.5%),
so this is not noise. A simple bandwidth ceiling would yield equal goodput at both sizes. Goodput
collapsing at the smaller size is the signature of **TCP incast**: synchronised senders overflow the
switch buffer and the transfer becomes retransmission-timeout-bound, whereas 1 MB flows run long
enough for TCP to recover and stream near line rate. Rivanna does not show the effect — its fabric has
deeper buffers and credit-based flow control.

**Implication for H1.** This identifies the linear gatherv as the specific scaling limiter on commodity
cloud networks, and motivates a tree-based gatherv for the serverless arm. I have deliberately *not*
changed it yet: the linear form is what makes Experiment B's FMI-vs-UCC comparison apples-to-apples,
and switching algorithms mid-study would break comparability with the data already collected. If we
adopt it, it should be behind a flag, off by default, and reported as a separate arm.

### 3.3 Why does allgather on Fargate break trend at N = 64?

Same mechanism, worse exposure. Allgather delivers all N contributions to *every* rank, so aggregate
volume is N× that of gather — at N=64 with 1 MB that is 64 MB per rank, ~4 GB in aggregate.

The anomaly is best seen as **throughput falling as messages get larger**, the opposite of the
expected amortisation of per-message overhead (Fargate, N=64):

| payload | 32 KB | 256 KB | 1 MB |
|---|---|---|---|
| throughput (MB/s) | 1.70 | 1.60 | **1.39** |

Corresponding latencies are 113 ms (N=16) → 291 ms (N=32) → 722 ms (N=64) at 1 MB. N = 64 is simply
where per-rank delivered volume passes what the 2-vCPU task network can sustain; past that point the
curve is set by the fabric, not by the algorithm.

### 3.4 Why does Fargate fluctuate so widely in Figure 1?

Here the data argues for a more careful claim than the figure suggests. Measured run-to-run
coefficient of variation across every collective, N, and payload:

| platform | median CV | p90 CV | max CV |
|---|---|---|---|
| Fargate | 10.1% | 23.3% | 56.0% |
| Rivanna | 5.6% | 20.4% | 61.0% |

Fargate is about **1.8× more variable at the median**, comparable in the tail, and Rivanna's worst case
is actually the higher of the two. The visual impression of wide fluctuation is amplified by two
plotting effects: log axes compress Rivanna's ~10× smaller absolute values, and equal *relative*
variance looks far larger in absolute milliseconds on the slower platform. I would not claim Fargate is
categorically unstable on this evidence.

The genuinely Fargate-specific sources of variance, ranked by expected contribution — these are
hypotheses, not yet isolated:

1. **Placement non-determinism.** Tasks land on arbitrary multi-tenant hosts, re-randomised every run,
   with no placement group or locality guarantee. Rivanna's scheduler allocates topologically
   adjacent nodes.
2. **Shared, burstable network at small task sizes.** We standardised on 2 vCPU to fit N = 64 inside the
   140-vCPU quota; smaller tasks sit in a lower network tier.
3. **Commodity TCP over a shared datacenter fabric** versus a dedicated HPC interconnect.

One artifact has already been eliminated: an earlier version of this sweep mixed 2- and 4-vCPU tasks,
which injected real variance into the comparison. The current figures come from the uniform 2-vCPU
re-run.

This connects directly to **Experiment C**, which predicts a CV reduction of ≥50% from
resource-deterministic scheduling. The CVs above are the baseline C has to improve on, and they
suggest the interesting variance to attack is on the cloud side. The natural follow-up is to isolate
hypothesis 1 by re-measuring with placement groups or pinned task placement — if placement dominates,
C's mechanism has a clear target.

---

## 4. What this leaves open

1. **Experiment E's Ray/Plasma arm** remains the critical path — the 4.2× figure is still against the
   HTTP diagnostic baseline only, which is the "strawman baseline" critique waiting to be made.
2. **The Cosmic AI end-to-end run** is the next step: the operators, the compiler, and the executor are
   in place and tested; the remaining work is the end-to-end harness over real AstroMAE outputs.
3. **Whether to adopt a tree gatherv** for the serverless arm (§3.2) — a measured decision, gated on
   whether establishment or transfer dominates at the target N, and on preserving Exp B comparability.
4. **Isolating the placement hypothesis** behind Fargate's variance (§3.4), which feeds Experiment C.
5. **Validating the rendezvous path on real Lambda.** Hole punching cannot be exercised from a single
   host — all local ranks share one public IP, so the NAT case is untestable locally. The rendezvous
   server is deployed and healthy; the remaining validation needs genuinely distinct hosts.