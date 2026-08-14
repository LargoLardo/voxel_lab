---
type: "query"
date: "2026-08-13T22:03:04.457783+00:00"
question: "Audit and redesign the genes; add paired counterfactual batches, stratified family/gene pool conditions, structural losses, and a shared-backbone family-adapter FiLM architecture."
contributor: "graphify"
outcome: "useful"
source_nodes: ["NeuralCA3D", "StatePool", "morphology_loss()", "rollout()", "checkpointing.py", "genomes.py", "targets_3d.py", "trainer.py"]
---

# Q: Audit and redesign the genes; add paired counterfactual batches, stratified family/gene pool conditions, structural losses, and a shared-backbone family-adapter FiLM architecture.

## Answer

Expanded from original query via graph vocab: genome, targets, pool, losses, training, neural, conditional, rollout, checkpointing, environment, dice, interpolation. Implemented tree genome schema v2 and target schema v3; removed taper; locked light tropism to environment training; added monotonic target audits, paired synchronized counterfactual rollouts, family/gene pool strata, structural and counterfactual losses, and TreeFamilyNCA3D with a shared perception backbone plus family FiLM and heads. Updated checkpoint conversion, lab, ecology, presets, and docs. CPU tests and RTX 4050 CUDA probes pass.

## Outcome

- Signal: useful

## Source Nodes

- NeuralCA3D
- StatePool
- morphology_loss()
- rollout()
- checkpointing.py
- genomes.py
- targets_3d.py
- trainer.py