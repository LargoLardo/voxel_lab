# Graph Report - voxels  (2026-08-09)

## Corpus Check
- 113 files · ~38,916 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 489 nodes · 1055 edges · 31 communities (25 shown, 6 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 31 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4614a849`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- EcologyWorld
- Ecology Scenario Suite
- StateLayout
- train
- LabSession
- NeuralCA2D
- ui.py
- make_target_3d
- Q: how can i run see and interact with the environments
- damage_3d
- StatePool
- Q: its erroring and saying: invalid request token; it does not let me place a seed nor play anything
- generate_report.py
- run_ecology_experiment.py
- run_full_experiment.py
- summarize_results.py
- morphovoxel/__init__.py
- morphovoxel
- Q: please produce a ui that i can use to navigate through all of these
- Q: please show how many steps are occuring per second, as well as an easier ui with sliders to adjust configs hi
- Q: How does the dashboard select CPU or CUDA and display live cellular automaton frames?
- Q: PermissionError replacing live.json during training preview
- Q: is  it possible to interact with each of  the completed runs in a design lab, such as placing a seed down with double click and then watching how the run's result would grow it accordingly and erasing parts using the mouse to see growth/regeneration. something similar to [https://distill.pub/2020/growing-ca/](https://distill.pub/2020/growing-ca/)
- Q: Review the smallest robust interactive checkpoint-backed Design Lab implementation
- morphovoxel/metrics.py
- Q: make an option to display the 3d voxel runs in the design lab as an actual 3d depiction with 3d cubes/voxel representations instead of 2d perspectives.
- trainer.py
- Q: pleaes make it
- Q: Inspect existing tests and add the smallest regression test for stale dashboard tokens after server restart.
- Q: Review the invalid request token bug and recommend the smallest secure restart-safe POST retry fix.
- Q: how do i run this

## God Nodes (most connected - your core abstractions)
1. `train()` - 42 edges
2. `StateLayout` - 32 edges
3. `load_config()` - 26 edges
4. `NeuralCA3D` - 26 edges
5. `main()` - 24 edges
6. `LabSession` - 24 edges
7. `seed_state()` - 24 edges
8. `EcologyWorld` - 19 edges
9. `one_hot_genomes()` - 19 edges
10. `main()` - 18 edges

## Surprising Connections (you probably didn't know these)
- `Experiment Report Research Question` --semantically_similar_to--> `Genome-Conditioned Local Morphogenesis Research Question`  [INFERRED] [semantically similar]
  reports/experiment_report.md → README.md
- `Matched Multi-Seed Evidence Requirement` --semantically_similar_to--> `Matched Multi-Seed Scientific Evidence`  [INFERRED] [semantically similar]
  reports/experiment_report.md → README.md
- `test_step_rate_uses_completed_updates()` --calls--> `steps_per_second()`  [EXTRACTED]
  tests/test_ui.py → morphovoxel/utils.py
- `test_live_preview_drops_locked_metadata_instead_of_crashing()` --calls--> `write_live_preview()`  [EXTRACTED]
  tests/test_ui.py → morphovoxel/utils.py
- `Phase 4 Regeneration Evaluation Configuration` --implements--> `Regeneration Evaluation`  [INFERRED]
  configs/phase4_regeneration.yaml → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **MorphoVoxel Phased Experiment Pipeline** — configs_full_experiment_full_pipeline, configs_phase1_2d_phase1_2d_config, configs_phase2_3d_phase2_3d_config, configs_phase3_conditional_phase3_conditional_config, configs_phase4_regeneration_training_phase4_regeneration_training_config, configs_phase4_regeneration_phase4_regeneration_config, configs_phase5_ecology_phase5_ecology_config [EXTRACTED 1.00]
- **Reduced Smoke Validation Suite** — configs_smoke_2d_smoke_2d_config, configs_smoke_3d_smoke_3d_config, configs_smoke_conditional_smoke_conditional_config, configs_smoke_ecology_smoke_ecology_config, configs_smoke_regeneration_smoke_regeneration_config, readme_smoke_test_limit [INFERRED 0.95]
- **Ecology Resource and Competition Scenarios** — configs_ecology_experiments_limited_water, configs_ecology_experiments_limited_light, configs_ecology_experiments_equal_distance_source, configs_ecology_experiments_unequal_distance_source, configs_ecology_experiments_shading, configs_ecology_experiments_uneven_water, readme_fixed_resource_ecology [INFERRED 0.95]

## Communities (31 total, 6 thin omitted)

### Community 0 - "EcologyWorld"
Cohesion: 0.08
Nodes (35): gate_growth(), Tensor, Local resource gains and biomass costs., Conservatively diffuse energy through each organism's local living neighborhood., transport_energy(), update_energy(), EcologyWorld, Tensor (+27 more)

### Community 1 - "Ecology Scenario Suite"
Cohesion: 0.05
Nodes (50): Abundant Resources Scenario, Different Genomes Scenario, Ecology Scenario Suite, Equal-Distance Water Source Scenario, Identical Genomes Scenario, Larger World Ecology Scenario, Limited Light Scenario, Limited Water Scenario (+42 more)

### Community 2 - "StateLayout"
Cohesion: 0.06
Nodes (45): load_checkpoint(), Any, Path, Reproducible training checkpoints., save_checkpoint(), main(), Normal-growth checkpoint evaluation., GenomeEncoder (+37 more)

### Community 3 - "train"
Cohesion: 0.08
Nodes (44): load_config(), Any, Path, YAML configuration helpers., Load a YAML mapping and reject ambiguous roots., save_config(), main(), Damage and regeneration evaluation. (+36 more)

### Community 4 - "LabSession"
Cohesion: 0.18
Nodes (6): inference_mode, LabSession, ndarray, Tensor, Return visible 3D cells for the browser's interactive cube renderer., One mutable cellular world driven by a trained local update rule.

### Community 5 - "NeuralCA2D"
Cohesion: 0.21
Nodes (10): NeuralCA2D, Tensor, Two-dimensional neural cellular automaton., Shared local 3x3 update rule for a semantic 2D state., perceive_2d(), Tensor, Fixed local 2D perception., Apply identity, Sobel-x/y, and Laplacian per channel. (+2 more)

### Community 6 - "ui.py"
Cohesion: 0.10
Nodes (33): BaseHTTPRequestHandler, HTTPStatus, Module, find_checkpoint(), _load_model_weights(), Path, Return the preferred local inference checkpoint for a run., Load only tensor weights from an app checkpoint, never arbitrary pickle code. (+25 more)

### Community 7 - "make_target_3d"
Cohesion: 0.21
Nodes (14): Deterministic procedural target library., _disk(), make_target_2d(), ndarray, Procedural 2D organism targets., Return deterministic occupancy and integer material maps., _segment(), _ball() (+6 more)

### Community 8 - "Q: how can i run see and interact with the environments"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: how can i run see and interact with the environments, Source Nodes

### Community 9 - "damage_3d"
Cohesion: 0.24
Nodes (8): damage_2d(), Tensor, Two-dimensional deletion damage., damage_3d(), Tensor, Three-dimensional deletion damage., Erase all channels in a geometric region; severity is world-volume based., test_damage_clears_all_channels_only_inside_mask()

### Community 10 - "StatePool"
Cohesion: 0.26
Nodes (6): PoolBatch, device, Tensor, Genome-safe pool of intermediate states., StatePool, test_pool_preserves_genome_pairings()

### Community 11 - "Q: its erroring and saying: invalid request token; it does not let me place a seed nor play anything"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: its erroring and saying: invalid request token; it does not let me place a seed nor play anything, Source Nodes

### Community 18 - "Q: please produce a ui that i can use to navigate through all of these"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: please produce a ui that i can use to navigate through all of these, Source Nodes

### Community 19 - "Q: please show how many steps are occuring per second, as well as an easier ui with sliders to adjust configs hi"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: please show how many steps are occuring per second, as well as an easier ui with sliders to adjust configs hi, Source Nodes

### Community 20 - "Q: How does the dashboard select CPU or CUDA and display live cellular automaton frames?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: How does the dashboard select CPU or CUDA and display live cellular automaton frames?, Source Nodes

### Community 21 - "Q: PermissionError replacing live.json during training preview"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: PermissionError replacing live.json during training preview, Source Nodes

### Community 22 - "Q: is  it possible to interact with each of  the completed runs in a design lab, such as placing a seed down with double click and then watching how the run's result would grow it accordingly and erasing parts using the mouse to see growth/regeneration. something similar to [https://distill.pub/2020/growing-ca/](https://distill.pub/2020/growing-ca/)"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: is  it possible to interact with each of  the completed runs in a design lab, such as placing a seed down with double click and then watching how the run's result would grow it accordingly and erasing parts using the mouse to see growth/regeneration. something similar to [https://distill.pub/2020/growing-ca/](https://distill.pub/2020/growing-ca/), Source Nodes

### Community 23 - "Q: Review the smallest robust interactive checkpoint-backed Design Lab implementation"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Review the smallest robust interactive checkpoint-backed Design Lab implementation, Source Nodes

### Community 24 - "morphovoxel/metrics.py"
Cohesion: 0.28
Nodes (13): _binary(), connected_components(), dice_score(), morphology_metrics(), paired_bootstrap_interval(), paired_permutation_test(), ndarray, Growth and regeneration metrics. (+5 more)

### Community 25 - "Q: make an option to display the 3d voxel runs in the design lab as an actual 3d depiction with 3d cubes/voxel representations instead of 2d perspectives."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: make an option to display the 3d voxel runs in the design lab as an actual 3d depiction with 3d cubes/voxel representations instead of 2d perspectives., Source Nodes

### Community 26 - "trainer.py"
Cohesion: 0.09
Nodes (33): plot_metrics(), Path, occupancy_image(), ndarray, Path, slice, Standalone 2D rendering., save_comparison() (+25 more)

### Community 27 - "Q: pleaes make it"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: pleaes make it, Source Nodes

### Community 28 - "Q: Inspect existing tests and add the smallest regression test for stale dashboard tokens after server restart."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Inspect existing tests and add the smallest regression test for stale dashboard tokens after server restart., Source Nodes

### Community 29 - "Q: Review the invalid request token bug and recommend the smallest secure restart-safe POST retry fix."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Review the invalid request token bug and recommend the smallest secure restart-safe POST retry fix., Source Nodes

### Community 30 - "Q: how do i run this"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: how do i run this, Source Nodes

## Knowledge Gaps
- **57 isolated node(s):** `morphovoxel`, `Answer`, `Outcome`, `Source Nodes`, `Answer` (+52 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `DashboardHandler` (6× useful, score=5.771363952) _(code changed — re-verify)_
- `DashboardServer` (5× useful, score=4.771249133) _(code changed — re-verify)_
- `LabSession` (3× useful, score=2.885165799)
- `rollout()` (3× useful, score=2.825455545)
- `train()` (3× useful, score=2.825012668) _(code changed — re-verify)_
- `_job_view()` (3× useful, score=2.825012668) _(code changed — re-verify)_
- `write_live_preview()` (3× useful, score=2.825012668)
- `run_ecology.py` (3× useful, score=2.823469251)
- `NeuralCA2D` (2× useful, score=1.885165895)
- `NeuralCA3D` (2× useful, score=1.885165895)

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `StateLayout` connect `StateLayout` to `trainer.py`, `train`, `LabSession`, `ui.py`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `LabSession` connect `LabSession` to `StateLayout`, `NeuralCA2D`, `ui.py`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `train()` connect `train` to `StateLayout`, `damage_3d`, `StatePool`, `morphovoxel/metrics.py`, `trainer.py`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **What connects `morphovoxel`, `Answer`, `Outcome` to the rest of the system?**
  _57 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `EcologyWorld` be split into smaller, more focused modules?**
  _Cohesion score 0.07616892911010557 - nodes in this community are weakly interconnected._
- **Should `Ecology Scenario Suite` be split into smaller, more focused modules?**
  _Cohesion score 0.052244897959183675 - nodes in this community are weakly interconnected._
- **Should `StateLayout` be split into smaller, more focused modules?**
  _Cohesion score 0.06164383561643835 - nodes in this community are weakly interconnected._