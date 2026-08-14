---
type: "query"
date: "2026-08-14T01:44:44.875549+00:00"
question: "CUDA OOM during tree regeneration persistence rollout"
contributor: "graphify"
outcome: "useful"
source_nodes: ["train()", "rollout()", "trainer.py"]
---

# Q: CUDA OOM during tree regeneration persistence rollout

## Answer

Expanded from graph vocabulary: [cuda, rollout, persistence, trainer, training, regeneration, pool, steps]. The trainer retains both growth and persistence rollout graphs for backward; batch 4 could sample 64+96=160 differentiable steps. Added a 96-step differentiable horizon cap to regeneration/environment presets while preserving no-gradient 512-step validation.

## Outcome

- Signal: useful

## Source Nodes

- train()
- rollout()
- trainer.py