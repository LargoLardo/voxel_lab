# MorphoVoxel

MorphoVoxel is a local artificial-life workbench for training and exploring neural cellular automata (NCAs). Its current focus is persistent 3D tree organisms: first learn one dependable specialist, then expand it into a shared model controlled by continuous genomes, regeneration training, and local environmental context.

## Quick start

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m morphovoxel.ui --open
```

On Linux or macOS, replace `.venv\Scripts\python` with `.venv/bin/python`.

The dashboard opens on `http://127.0.0.1:8765`. It lists the useful full presets first and smoke checks last. Use it to launch training, follow logs and completed-rollout previews, edit genomes and environments, inspect targets, interact with checkpoints in 3D, validate candidates, and browse admitted variants. View Checkpoints supports seed placement, play/pause/single-step, reset, and erase/damage tools.

Preview disposable test, cache, coverage, and build artifacts, then remove them explicitly:

```powershell
.venv\Scripts\python -m morphovoxel.cleanup
.venv\Scripts\python -m morphovoxel.cleanup --apply
```

Cleanup never enters `.venv`, `runs`, `variant_archive`, or `graphify-out`, so environments, checkpoints, archived organisms, and Graphify data are preserved.

## Recommended tree pipeline

Start with the specialist. This is the first useful training command:

```powershell
.venv\Scripts\python -m morphovoxel.train_3d --config configs\tree_specialist.yaml
```

To run all five stages from scratch instead, use:

```powershell
.venv\Scripts\python scripts\run_full_experiment.py --config configs\full_experiment.yaml
```

The command runs these stages in order and stops on the first failure:

1. `tree_specialist.yaml` learns one default branching tree.
2. `tree_family.yaml` initializes from the specialist's `best.pt`, adds continuous genome and environment inputs without changing the copied behavior initially, then trains a balanced continuous-genome curriculum in one fixed environment.
3. `tree_regeneration.yaml` resumes the exact family architecture and trains on damaged mature pool states.
4. `tree_environment.yaml` resumes the regeneration checkpoint and trains across randomized local conditions.
5. `tree_ecology.yaml` loads the environment-trained family and places two semantic tree genomes in one resource field.

The individual commands are:

```powershell
.venv\Scripts\python -m morphovoxel.train_3d --config configs\tree_specialist.yaml
.venv\Scripts\python -m morphovoxel.train_conditional --config configs\tree_family.yaml
.venv\Scripts\python -m morphovoxel.train_conditional --config configs\tree_regeneration.yaml
.venv\Scripts\python -m morphovoxel.train_conditional --config configs\tree_environment.yaml
.venv\Scripts\python -m morphovoxel.run_ecology --config configs\tree_ecology.yaml
```

Do not skip a prerequisite unless you replace its checkpoint path with a compatible checkpoint. Loading mismatched model shapes is rejected rather than silently reinterpreted.

## Smoke checks

These CPU presets are independent, intentionally tiny pipeline checks. They do not produce useful organisms or stable checkpoints.

```powershell
.venv\Scripts\python -m morphovoxel.train_3d --config configs\smoke_tree_specialist.yaml
.venv\Scripts\python -m morphovoxel.train_conditional --config configs\smoke_tree_family.yaml
.venv\Scripts\python -m morphovoxel.train_conditional --config configs\smoke_tree_regeneration.yaml
.venv\Scripts\python -m morphovoxel.train_conditional --config configs\smoke_tree_environment.yaml
.venv\Scripts\python -m morphovoxel.run_ecology --config configs\smoke_tree_ecology.yaml
```

## Three separate inputs

- **Genome** is inherited and fixed for an organism unless live remodeling is explicitly enabled. A tree genome contains one discrete topology family, nine bounded continuous style genes, and one reproducible style seed. Taper was removed; light tropism is locked until directional-light environment training.
- **Environment** changes around an organism. The NCA can receive local light, water, energy, substrate, obstacles, neighboring occupancy, gravity, and wind fields before proposing its update.
- **Cell state** is developmental memory: occupancy, material logits, optional energy, and hidden channels. Damage clears every state channel in the affected region.

The update rule is conceptually:

```text
next_state = NCA(perception(cell_state), organism_genome, local_environment)
```

Legacy one-hot genomes remain loadable, but they are categorical selectors. A midpoint between two one-hot labels was not trained as interpretable DNA. Continuous tree genes are different: random samples, within-family interpolations, and bounded mutations are paired with deterministic procedural targets during training.

## Specialists, families, and ecology

Use a specialist while inventing or stabilizing one organism: it dedicates all model capacity to one target and isolates failures. Use a shared family model when compact inference, named variations, interpolation, or mutation matter. The supported workflow starts with a specialist and expands its compatible weights into the family model.

Ecology can route either one shared checkpoint with different genomes or separate specialist checkpoints by organism. Sharing a world only creates mechanical competition for occupancy and resources. It does not create learned tropism, cooperation, or competition unless the participating model was trained with the corresponding neighbor and resource context; `tree_environment.yaml` is the relevant shared-family stage.

## Persistence and the Variant Archive

Family training uses stratified low/high counterfactual pairs: seed, style, environment, fire masks, and damage are shared while exactly one gene changes. The loss combines balanced occupancy/material terms with soft Dice/IoU, distance-to-target, height, width, volume, centroid, and branch-distribution descriptors. The model uses one shared perception backbone with family-specific FiLM and output heads. Living masks, magnitude/range penalties, gradient clipping, and non-finite checks remain active.

`latest.pt` is the final optimizer state. `best.pt` is updated when the configured deterministic validation panel matches or improves its worst-case score, so an early zero-score tie cannot freeze the pipeline at its first validation window. A checkpoint is not stable merely because it is named `best.pt`; inspect its validation report and require `accepted: true`. Full tree presets validate for at least 256 steps and include recovery trials. Archive admission is stricter: the default minimum is 512 growth/persistence steps plus 128 recovery steps across fixed stochastic and environmental cases.

Procedural tree targets use target schema version 3 and tree genomes use schema version 2. Earlier schemas are deliberately rejected because the gene count, target geometry, and family architecture changed. Retrain specialist → family → regeneration → environment rather than resuming an old tree checkpoint.

A “new variant” means a new valid genome/style-seed combination, not proof of a fundamentally new species. Mutation and interpolation stay inside the declared gene bounds, and interpolation is allowed only within one discrete family. A candidate outside the sampled training distribution can still fail; archive admission requires finite, bounded, connected, persistent, and regenerative validation rather than visual appeal alone.

## GPU guidance

`device: auto` selects CUDA when the installed PyTorch build can use it and otherwise falls back to CPU. The full presets use FP32 and a `16³` world; family uses batch 8, while the longer regeneration/environment horizons use batch 4 on an RTX 4050 Laptop GPU with 6 GB VRAM. Long 256–512-step validation runs under no-gradient inference.

A focused probe of the redesigned family model on the target RTX 4050 (5.997 GiB usable, PyTorch 2.13.0+cu126) completed batch 8 at 48 growth + 32 persistence steps, all structural/counterfactual losses, backward, clipping, and Adam in 1.059 seconds. It peaked at 3088.5 MiB allocated / 3276.0 MiB reserved. Regeneration and environment presets cap their retained differentiable horizon at 96 steps, avoiding allocator-fragile 64 + 96 step graphs; their 512-step validation still runs under no-gradient inference. Close other GPU-heavy applications before full training; these probes do not predict convergence time or morphology quality.

For a `24³` world, start with batch size 1 or 2. If CUDA runs out of memory, reduce batch size first, then rollout/persistence length, hidden channels, or model width. Do not enable mixed precision until the model remains finite and bounded in FP32. The five full stages contain 17,000 optimizer iterations plus repeated multi-case persistence/recovery panels; on a thermally constrained laptop, budget multiple hours and be prepared for an overnight run. Panel validation can dominate wall time even though its no-gradient memory use is modest. Run one stage at a time if you need dependable checkpoints around reboots or thermal limits. Smoke presets usually finish in seconds or minutes and are the fast installation check.

Install a CUDA-enabled PyTorch wheel using the current command from the [official PyTorch selector](https://pytorch.org/get-started/locally/). Verify CUDA before a full run:

```powershell
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Legacy presets

The older `phase1_2d.yaml` through `phase5_ecology.yaml`, conditional one-hot experiments, regeneration sweeps, and `ecology_experiments.yaml` are retained for checkpoint compatibility and comparison. They are legacy workflows and are not prerequisites for the tree pipeline. Their smoke variants remain available at the end of the dashboard preset list.

Useful legacy and analysis commands include:

```powershell
.venv\Scripts\python -m morphovoxel.visualize --run-dir runs\<run_name>
.venv\Scripts\python scripts\run_ecology_experiment.py --config configs\ecology_experiments.yaml
.venv\Scripts\python scripts\summarize_results.py --runs-root runs
.venv\Scripts\python scripts\generate_report.py --run-dir runs\<run_name>
```

## Outputs and further reading

Each `runs/<name>` directory stores its exact YAML snapshot, versioned checkpoint metadata, CSV metrics, checkpoints, rollout arrays, targets, and visualizations. Training live preview atomically replaces one image after every tenth completed iteration; it does not save every cellular step. Existing runs are never needed to launch the dashboard and should not be deleted merely to start a new run.

See [docs/architecture.md](docs/architecture.md) for the model boundary, curriculum, validation, archive, ecology routing, and compatibility rules. The old planned-study framing remains in [reports/experiment_report.md](reports/experiment_report.md) as a clearly labeled legacy document.

Design references:

- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- [Goal-Guided Neural Cellular Automata](https://arxiv.org/abs/2205.06806)
- [Neural Cellular Automata Manifold](https://openaccess.thecvf.com/content/CVPR2021/html/Hernandez_Neural_Cellular_Automata_Manifold_CVPR_2021_paper.html)
- [Growing 3D Artefacts and Functional Machines with Neural Cellular Automata](https://arxiv.org/abs/2103.08737)
