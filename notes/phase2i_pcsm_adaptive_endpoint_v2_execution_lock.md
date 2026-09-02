# PCSM adaptive-endpoint v2 execution lock

Frozen before any v2 model run and after a metric-free MultiRC v1
infrastructure interruption.  The interrupted run produced no durable
internal/development/confirmation metric.  This lock changes peak-memory
execution only; the scientific endpoint rule remains fixed by
`phase2i_pcsm_adaptive_endpoint_v2_protocol.md` (SHA-256
`C0BB4F10D17FE10AFB30FDB499549492A2191D8E391ED3544E8610A34B51B040`).

## Logical training trajectory

- logical paired batch size: 4 (two True and two False in the exact frozen v1
  order);
- physical training microbatch size: 2, using the consecutive slices `[0:2]`
  and `[2:4]` of each logical batch;
- for a microbatch of size `m`, backpropagate its mean weighted PCSM loss times
  `m / 4`; zero gradients once before both microbatches, clip once after both,
  and call AdamW `step()` once;
- 384 optimizer steps, 1,536 total example exposures, 768 exposures per class,
  and two complete fit cycles;
- ordered logical exposure IDs and their SHA-256 must exactly match the frozen
  v1 schedule;
- learning rate 3e-4, weight decay zero, dropout disabled, and every other
  scientific training field remain unchanged.

This is algebraically the same logical-batch objective, but different physical
batching can change floating-point accumulation.  Results must be labelled an
execution-v2 trajectory and must not be described as bitwise identical to v1.

## Evaluation and durability

- evaluation microbatch size: 2 for free generation, constrained-label
  scoring, and margin evaluation; no audit row may be skipped or reordered;
- thresholds and metrics are computed only after concatenating the complete
  ordered partition outputs;
- after step 384 and before post-training internal evaluation, atomically write
  the compact adapter, resource ledger, exposure SHA, and a
  `trajectory_completed_pending_internal_eval` lock;
- intermediate recovery artifacts, if implemented, are written every 32
  logical steps and contain adapter/optimizer state, completed step, and the
  exposure-prefix SHA.  They are `recovery_only`, never evaluation- or
  selection-eligible, and must be removed from scientific candidate counts;
- a resume is allowed only after all source/model/protocol/code hashes match,
  no later audit partition has been accessed, and the exact next logical step
  is reconstructed.  Resume cannot repeat or add an exposure or optimizer
  step.

The only selection-eligible checkpoint remains the predeclared step-384
adapter.  Resource controls cannot change the endpoint rule, threshold fitter,
data split, checkpoint count, or any gate.
