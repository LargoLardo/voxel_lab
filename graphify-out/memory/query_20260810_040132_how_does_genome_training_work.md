---
type: "query"
date: "2026-08-10T04:01:32.512502+00:00"
question: "how does genome training work"
contributor: "graphify"
outcome: "useful"
source_nodes: ["one_hot_genomes()", "train()", "NeuralCA3D", "morphology_loss()", "Phase 3 Conditional Training Configuration"]
---

# Q: how does genome training work

## Answer

Expanded tokens: [genome, conditional, training, target, morphology, shared, perception, rollout, loss, occupancy, material, checkpoint]. In Phase 3, four fixed morphology labels are encoded as four-element one-hot vectors. With batch size four, each iteration cycles through all four labels and pairs each with its procedural target. The same 3D NCA weights process every sample; the selected genome vector is broadcast to every voxel and concatenated with local perception features. The model rolls forward for 24-48 stochastic steps, then backpropagation minimizes occupancy, material, leakage, and extra stability losses. This is conditional multi-task learning, not mutation, crossover, or evolutionary search. Phase 4 retains this genome mechanism but trains from a state pool with random damage for regeneration.

## Outcome

- Signal: useful

## Source Nodes

- one_hot_genomes()
- train()
- NeuralCA3D
- morphology_loss()
- Phase 3 Conditional Training Configuration