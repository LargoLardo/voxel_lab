---
type: "query"
date: "2026-08-11T03:20:34.512873+00:00"
question: "Implement the Goal Mode checkpoint layer with versioned metadata, legacy compatibility, incompatibility errors, and specialist-to-family NeuralCA3D conversion"
contributor: "graphify"
outcome: "useful"
source_nodes: ["checkpointing.py", "save_checkpoint()", "load_checkpoint()", "NeuralCA3D"]
---

# Q: Implement the Goal Mode checkpoint layer with versioned metadata, legacy compatibility, incompatibility errors, and specialist-to-family NeuralCA3D conversion

## Answer

Expanded from original query via graph vocab: [checkpoint, checkpointing, model, neural, genome, environment, context, metadata, payload, persistence, weights, load]. The graph identified morphovoxel/checkpointing.py as the shared save/load boundary used by trainer, evaluation, and NeuralCA3D workflows. The additive implementation keeps old payload keys, adds explicit versioned metadata, validates declared schemas and tensor shapes, and converts a specialist NeuralCA3D by copying compatible weights while zero-initializing added genome/environment inputs.

## Outcome

- Signal: useful

## Source Nodes

- checkpointing.py
- save_checkpoint()
- load_checkpoint()
- NeuralCA3D