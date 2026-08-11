---
type: "query"
date: "2026-08-11T03:23:06.020217+00:00"
question: "Implement ecology model routing and pre-update local environment context while preserving legacy ecology"
contributor: "graphify"
outcome: "useful"
source_nodes: ["EcologyWorld", "ecology_step()", "NeuralCA3D", "run_ecology.py"]
---

# Q: Implement ecology model routing and pre-update local environment context while preserving legacy ecology

## Answer

Expanded from the request via graph vocabulary: [ecology, model, genome, context, environment, organism, neighbor, occupancy, shared, specialized, world, step]. EcologyWorld now carries optional obstacles and per-organism model ids; ecology_step computes current light and transported energy before proposals, builds the 12-channel local context, and uses ModelRouter for either one shared continuous-genome model or grouped specialist models while restoring organism order. Legacy models without context metadata retain the old no-context call path. Focused tests and a two-step CLI smoke run passed.

## Outcome

- Signal: useful

## Source Nodes

- EcologyWorld
- ecology_step()
- NeuralCA3D
- run_ecology.py