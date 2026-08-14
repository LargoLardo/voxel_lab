---
type: "query"
date: "2026-08-14T02:08:40.929435+00:00"
question: "Why did bigworldgenome_real run out of CUDA memory?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["train()", "rollout()", "trainer.py"]
---

# Q: Why did bigworldgenome_real run out of CUDA memory?

## Answer

Expanded from graph vocabulary: [cuda, rollout, persistence, trainer, training, steps]. The saved job configuration used a 32-cubed tree-family world with batch 8, versus the standard 16-cubed family world. The trainer retains growth and persistence graphs for backward. A CUDA guard now rejects this known-unworkable small-GPU setup and specifies batch 2 plus a 48-step differentiable cap for 32-cubed training.

## Outcome

- Signal: useful

## Source Nodes

- train()
- rollout()
- trainer.py