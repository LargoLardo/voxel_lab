---
type: "debugging"
date: "2026-08-13T02:25:18.758685+00:00"
question: "Why did regeneration alternate between 0.83 and exact-zero loss, and why was tree-family output insensitive to genomes?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["make_tree_target", "sample_family_data", "morphology_loss", "StatePool", "train"]
---

# Q: Why did regeneration alternate between 0.83 and exact-zero loss, and why was tree-family output insensitive to genomes?

## Answer

Version-1 tree targets used a half-voxel base but an integer connectivity seed, producing empty thin-tree targets. Dead states paired with empty targets have exact-zero loss; dead states with microscopic targets produce the 0.8318 material-entropy plateau. Global pool selection treated them as easy and propagated their identities. In addition, all initial family entries were branching and strict zero-score ties froze best.pt at step 250, so downstream stages used early genome-insensitive checkpoints. Fix with target schema v2 integer base, nonempty invariants, dead-state reseeding, viability-preserving damage, foreground-balanced occupancy loss, balanced/preserved family sampling, fixed family-stage environment, tied-score checkpoint advancement, and resume pool expansion.

## Outcome

- Signal: useful

## Source Nodes

- make_tree_target
- sample_family_data
- morphology_loss
- StatePool
- train