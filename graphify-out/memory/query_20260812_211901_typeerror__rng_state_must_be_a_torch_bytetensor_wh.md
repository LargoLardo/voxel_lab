---
type: "query"
date: "2026-08-12T21:19:01.709280+00:00"
question: "TypeError: RNG state must be a torch.ByteTensor when resuming train_conditional on CUDA"
contributor: "graphify"
outcome: "useful"
source_nodes: ["trainer.py", "train", "checkpointing.py"]
---

# Q: TypeError: RNG state must be a torch.ByteTensor when resuming train_conditional on CUDA

## Answer

Expanded from original query via vocab: [checkpoint, cuda, random, state, generator, trainer, training]. The trainer loads checkpoints with map_location=cuda, which also places saved RNG state tensors on CUDA. torch.cuda.set_rng_state_all requires CPU uint8 ByteTensors. The shared _restore_rng_state path now normalizes CPU and CUDA RNG payload values to contiguous CPU torch.uint8 tensors before restoring them. A focused regression and an RTX 4050 reproduction with deliberately CUDA-mapped RNG tensors both pass.

## Outcome

- Signal: useful

## Source Nodes

- trainer.py
- train
- checkpointing.py