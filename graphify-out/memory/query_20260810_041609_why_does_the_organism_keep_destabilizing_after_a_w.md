---
type: "query"
date: "2026-08-10T04:16:09.828722+00:00"
question: "why does the organism keep destabilizing after a while when i run it on genome or regeneration and i test it, theres no persistence. https://distill.pub/2020/growing-ca/ use this as a source if helpful"
contributor: "graphify"
outcome: "useful"
source_nodes: ["train()", "stability_loss()", "StatePool", "NeuralCA3D", "Phase 3 Conditional Training Configuration", "Phase 4 Regeneration Training Configuration"]
---

# Q: why does the organism keep destabilizing after a while when i run it on genome or regeneration and i test it, theres no persistence. https://distill.pub/2020/growing-ca/ use this as a source if helpful

## Answer

Expanded tokens: [persistence, stability, regeneration, pool, damage, rollout, finite, loss, training, state, hidden, target]. The checkpoints learn an endpoint trajectory rather than a robust target attractor. Phase 3 trains only 24-48 rollout steps and eight continuation steps, always from fresh seeds. The stability loss compares only clamped occupancy, while morphology loss also clamps occupancy; raw occupancy and hidden/material channels can therefore drift far outside their intended range without proportional penalty. A direct branching-genome probe found the normal checkpoint raw maximum growing from 35.8 at step 16 to 17119 at step 256; regeneration drifted more slowly from 21.1 to 67.7. The model also gates deltas by a living mask but does not zero dead-cell state, unlike the referenced Distill alive-mask strategy. Phase 4 has a pool and damage, but samples/damages randomly rather than prioritizing mature low-loss states and reseeding the worst state. Finally, best.pt is only a copy of latest.pt, not a persistence-selected checkpoint.

## Outcome

- Signal: useful

## Source Nodes

- train()
- stability_loss()
- StatePool
- NeuralCA3D
- Phase 3 Conditional Training Configuration
- Phase 4 Regeneration Training Configuration