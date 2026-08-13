---
type: "query"
date: "2026-08-13T02:16:26.770174+00:00"
question: "Fix family sampling so narrow-span initialization does not collapse to one tree family, while keeping targets nonempty."
contributor: "graphify"
outcome: "useful"
source_nodes: ["genomes.py", "targets_3d.py", "trainer.py", ".sample()"]
---

# Q: Fix family sampling so narrow-span initialization does not collapse to one tree family, while keeping targets nonempty.

## Answer

Expanded from the original query via graph vocabulary: [genome, genomes, target, targets, sample, random, initial, equal, mode, mixing, hybrids, conditional]. The traversal connected genome encoding, target generation, StatePool sampling, and trainer flow. The focused fix balances all TREE_FAMILIES in deterministic shuffled initial batches, permits random replacement identities from every family even at narrow spans, rejects empty target occupancy, and reflects inherited mutations instead of clipping genes at boundaries.

## Outcome

- Signal: useful

## Source Nodes

- genomes.py
- targets_3d.py
- trainer.py
- .sample()