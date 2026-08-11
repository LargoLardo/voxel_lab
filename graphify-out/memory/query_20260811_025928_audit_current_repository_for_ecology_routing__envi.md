---
type: "query"
date: "2026-08-11T02:59:28.564341+00:00"
question: "Audit current repository for ecology routing, environment inputs, configs, checkpoint compatibility, docs, CUDA verification, and tests."
contributor: "graphify"
outcome: "useful"
source_nodes: ["EcologyWorld", "ecology_step()", "run_ecology.py", "checkpointing.py", "DashboardHandler", "Shared versus Specialized NCA Comparison", "NeuralCA3D"]
---

# Q: Audit current repository for ecology routing, environment inputs, configs, checkpoint compatibility, docs, CUDA verification, and tests.

## Answer

Expanded from original query via graph vocab: [ecology, shared, specialized, environment, context, configs, checkpoint, metadata, cuda, device, tests, dashboard]. Graph traversal identified EcologyWorld, ecology_step, run_ecology, checkpointing, DashboardHandler, tests, and README. Direct source verification found one shared ecology model and post-proposal resource handling; current uncommitted groundwork adds EnvironmentSpec and NeuralCA3D context ingress, but ecology does not yet feed dynamic context or route specialist models. Legacy full-first and smoke-last ordering is tested, while requested tree presets are absent. Checkpoints lack explicit format/model/genome/environment/target schema metadata and legacy compatibility fixtures. README is partly reframed but no architecture design document exists. A transient RTX 4050 FP32 probe at world 16 batch 4 completed 80 differentiable steps at 748.5 MiB peak allocated and 256 inference steps at 13.2 MiB, finite; checked-in verification is CPU-only and optional CUDA coverage is absent.

## Outcome

- Signal: useful

## Source Nodes

- EcologyWorld
- ecology_step()
- run_ecology.py
- checkpointing.py
- DashboardHandler
- Shared versus Specialized NCA Comparison
- NeuralCA3D