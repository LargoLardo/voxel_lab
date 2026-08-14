---
type: "query"
date: "2026-08-14T01:01:46.956931+00:00"
question: "the regeneration part of the pipeline is stuck at 250 iterations beyond the previous checkpoint and doesnt proceed, please check why"
contributor: "graphify"
outcome: "useful"
source_nodes: ["train()", "trainer.py", "rollout()", "checkpointing.py", "StatePool"]
---

# Q: the regeneration part of the pipeline is stuck at 250 iterations beyond the previous checkpoint and doesnt proceed, please check why

## Answer

Expanded from original query via graph vocab: regeneration, validation, checkpoint, persistence, rollout, trainer, training, cuda, pool, steps. Live evidence showed PID 3376 healthy and using CUDA. The resumed checkpoint was step 8000; at step 8250 validation_every triggered a 168-case panel (21 genome cases x 2 environments x 4 fire seeds), each with 512 validation plus 128 recovery steps, totaling 107,520 NCA updates. The trainer logged nothing during the panel, making it appear stuck. It completed and resumed at step 8251, progressing beyond 8300. Added on_trial progress callbacks and approximately 5-percent validation progress logs; focused tests pass.

## Outcome

- Signal: useful

## Source Nodes

- train()
- trainer.py
- rollout()
- checkpointing.py
- StatePool