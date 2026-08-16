# TabICLv2 implementation notes

Research snapshot: **2026-08-15**.

## Paper

[TabICLv2: A better, faster, scalable, and open tabular foundation model](https://arxiv.org/abs/2602.11139)
introduces three main advances over TabICL:

1. a much more diverse synthetic graph prior;
2. repeated feature grouping, early target-aware embeddings, and QASSMax;
3. a three-stage Muon pretraining recipe that grows the row context.

The architecture has complexity `O(n² + nm²)` for `n` rows and `m` features.
It first compresses each column with induced attention, collapses the feature
axis into a row embedding, then performs ICL across rows. The published
classification and regression models use width 128, 3 column blocks with 128
inducing tokens, 3 row blocks, 4 CLS tokens, and 12 ICL blocks at width 512.
Regression predicts 999 quantiles.

The published curriculum is:

- stage 1: 500K steps at 1,024 rows;
- stage 2: 40K steps at 400–10,240 rows;
- stage 3: 10K steps at 400–60,000 rows;
- batch size 64, AMP throughout, checkpointing beyond 20K rows, and
  FlashAttention-3 in stages 2–3.

This repository implements the architecture and a deliberately compact graph
prior. It does not claim to reproduce that full compute budget.

## Latest official code inspected

- Repository: [soda-inria/tabicl](https://github.com/soda-inria/tabicl)
- Main commit inspected: `bd030d47ff7d9f987f0f33431d743a3003bfe0d3`
  (2026-08-12)
- The July merge of
  [PR #135](https://github.com/soda-inria/tabicl/pull/135) added the public
  TabICLv2 graph prior, pretraining CLI, Muon optimizer, quantile regression,
  and three-stage scripts.
- The minimal official architecture was cross-checked against
  [soda-inria/nanotabicl](https://github.com/soda-inria/nanotabicl).

## Recent PRs that affected this design

- [#144: backend-agnostic device handling](https://github.com/soda-inria/tabicl/pull/144)
  was open and mergeable, updated 2026-08-13. Its real-hardware tests found MPS
  AMP useful for roughly 2× lower activation memory but no significant MPS
  speedup over CPU on an M4. Investigation of virtualized Apple runners isolated
  incorrect results to biased `F.linear` on rank-3 tensors
  ([PyTorch #192934](https://github.com/pytorch/pytorch/issues/192934)). This
  project therefore adds Linear bias separately and keeps MPS AMP opt-in.
- [#145: cgroup-aware CPU memory](https://github.com/soda-inria/tabicl/pull/145)
  was open. It matters to disk/CPU offloading in container environments. This
  micro implementation does not yet implement the official inference offloader,
  so it bounds cloud jobs through profiles instead.
- [#142: replace NaN-only checks with finite checks](https://github.com/soda-inria/tabicl/pull/142)
  was open. The compact prior uses `nan_to_num`, and estimators reject non-finite
  real input.
- [#141: Windows tutorial encoding](https://github.com/soda-inria/tabicl/pull/141)
  merged 2026-08-12 and is the latest main commit in the inspected checkout.

These PRs were read as design signals, not copied wholesale. Open PR behavior
may change before merge.

## Deliberate differences

- The graph prior has eight scalar random-function families, but omits the
  official prior's full correlated hyperparameter sampler, matrix-valued nodes,
  ExtraTrees bootstrap filter, and detailed converter system.
- Laptop profiles use AdamW by default because short, small-batch Muon runs can
  be unstable; cloud profiles expose Muon.
- Many-class mixed-radix ensembling, hierarchical classification, inference KV
  caching, and disk offloading are out of scope for this first micro build.
- The estimator supports at most the checkpoint's pretraining class count
  (10 for all included profiles).
