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

### macOS

From the repository directory, install and launch with:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m morphovoxel.ui --open
```

On an Apple Silicon Mac, `device: auto` selects the Apple GPU through PyTorch's
[Metal (MPS) backend](https://docs.pytorch.org/docs/stable/notes/mps.html).
The dashboard also offers **Apple GPU (Metal)** explicitly. Use a current native
ARM64 Python/PyTorch installation; the Metal training, checkpoint resume,
validation, and ecology paths are tested with PyTorch 2.14 on macOS 26.6.
`device: cpu` works when Metal is unavailable. Smoke presets intentionally select
CPU; choose Auto or Apple GPU in the dashboard to exercise Metal.

The first full training stage can also be launched directly:

```sh
.venv/bin/python -m morphovoxel.train_3d --config configs/tree_specialist.yaml
```

For the remaining commands below, use `.venv/bin/python` and forward slashes in
paths, such as `configs/tree_family.yaml`. Metal uses FP32 for model computation;
exact environment metadata stays on CPU. Checkpoints preserve the Metal random
stream. Metal training uses seeded randomness without requesting deterministic
GPU algorithms, since indexed-gradient accumulation on MPS lacks a deterministic
implementation. A startup message and run metadata record this mode. Repeated
Metal runs and runs across CPU, CUDA, and Metal are not guaranteed to be identical.

The dashboard opens on `http://127.0.0.1:8765`. It lists the useful full presets first and smoke checks last. Use it to launch training, follow logs and completed-rollout previews, edit genomes and environments, inspect targets, interact with checkpoints in 3D, validate candidates, and browse admitted variants. View Checkpoints supports seed placement, play/pause/single-step, reset, and erase/damage tools.

The header separates **saved runs** (experiment folders, including stopped runs) from **active jobs** (training processes). On macOS and Linux, restarting the dashboard reconnects surviving jobs to their logs, previews, and Stop buttons. Training keeps running during the restart. A recovered job that later exits is labeled **ended**, since its exit code is unavailable to the new dashboard process; check its log and saved results. An intentional Stop is labeled **stopped**.

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
2. `tree_family.yaml` initializes from a specialist or family checkpoint, learns the four basic families, then introduces gene and style variation in one fixed environment. Either part can also run separately.
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

In the dashboard's **02 Tree Family** page, choose **Basic families + variation**, **Learn the basic families only**, **Learn variation only**, **Learn live family transitions**, or **Learn live gene transitions**, then choose **Start from checkpoint**. All choices accept compatible specialist and Phase 2 checkpoints. Prefer a checkpoint that already grows all four families for family transitions, and one trained on variation for gene transitions. For example: specialist → basic families → variation → family transitions → gene transitions → regeneration. Use a new run name for each handoff.

The same options work in YAML:

```yaml
family_curriculum: variation  # full, basics, variation, transition, or gene_transition
initialize_from_checkpoint: runs/my_basic_families/checkpoints/best.pt
iterations: 8000
```

`initialize_from_checkpoint` copies model weights, converting a specialist when needed, and starts a fresh optimizer, pool, and curriculum. It is mutually exclusive with `resume` and the older `initialize_from_specialist` option. Use `resume` to continue the same curriculum with its saved training state; use `initialize_from_checkpoint` to switch curricula. Channel counts, hidden width, and schema versions must remain compatible.

The full curriculum allocates the first 25% of updates to basic families (`basic_family_fraction: 0.25`): all genes neutral, fixed environment, and `family_style_seeds: [0, 970806, 1941611, 2912417]`. These seeds span roughly quarter turns of the style phase and produce four distinct targets per family at 16³; the previous adjacent seeds produced identical targets. It validates all four families across those styles and the configured growth fire seeds, including long growth and recovery checks. The boundary checkpoint is saved as `basic_families.pt`. The transition follows the configured schedule, not a passing validation gate; inspect the report before treating a checkpoint as stable. Basics-only uses the entire update budget for this part.

Variation-only uses the entire budget for variation; full uses its remaining budget. Variation begins with single-gene deviations near ±0.15, widening to ±1 at 45% of the variation budget. After the first 25% (`combination_start_fraction`), other genes and random style seeds gradually enter the examples. Neutral pairs are sampled with probability 25% (`neutral_fraction`) throughout variation to help preserve the basic shapes. All weights remain trainable. `best.pt` comparisons restart when full training switches to variation, because scores on the two validation panels are not comparable. Logs record `curriculum_stage`, gene ranges, and sampling fractions.

Routine pool refreshes select the oldest sampled pairs, so every healthy family/gene condition receives updated curriculum examples. Dead pairs are still reseeded immediately. Accepted targets are reused when assembling a batch; identical neutral pairs generate their target once.

`family_curriculum: transition` teaches live remodeling between families. About 75% of batches grow source organisms for at least `transition_source_steps: 128` total steps, then change only their family input, retaining every occupancy, material and hidden-state channel. The destination is sampled from the other three families. The usual `rollout_steps` train the switch and `persistence_steps` train retention of the new shape. Source preparation runs without gradients; the trainable rollout starts at the switch. Source states are reused in the pool without overwriting their identity with switched states. The remaining 25% of batches rehearse ordinary growth and persistence to limit forgetting.

This first transition curriculum uses neutral genes, the configured basic style seeds, and a fixed environment. It teaches all 12 directed family changes, not arbitrary simultaneous gene/environment changes or repeated switches within one training rollout. Validation grows each source from a seed, switches it without resetting, then checks destination shape, persistence and damage recovery. A poor source shape also fails validation. With four styles and two fire seeds this is 96 trials, so validation costs more than basics. The CSV records source/destination identity and source IoU. The saved `growth.gif` demonstrates branching → weeping; `rollouts/transition.json` records the switch. In the Tree Genome Lab, load the resulting checkpoint, grow a tree, enable **Live remodel this mature organism**, choose another family and apply it. The existing `full` schedule remains basics → variation; run transitions as a separate checkpoint handoff.

`family_curriculum: gene_transition` teaches live edits to the eight Phase 2 shape genes, such as height and canopy spread. Fixed-genome variation teaches how to grow a tall tree; this curriculum also teaches an already-grown short tree to become tall, and the reverse. It uses the variation schedule: narrow gene ranges first, then wider ranges, varied background genes and more styles. Each nonneutral pair differs in just one gene; its two values are sampled inside the current range so small and large edits both remain in training. Family, style and environment stay fixed within an edit. Light tropism remains part of environment training.

About 75% of batches mature the paired source trees, then swap their gene inputs and targets while preserving their complete states. This teaches both directions using targets and distance maps already in the pool. Source preparation grows only young entries to `transition_source_steps`; mature entries skip that work. It runs without gradients, while switching and subsequent persistence are trained normally. The other 25% of batches rehearse fixed-genome growth. Neutral pairs remain in the sampling mix as no-change practice. Initialize from a variation checkpoint with matching state channels and hidden layers; use `resume` only to continue the same curriculum.

Gene-transition validation runs 128 cases: four families × eight genes × two edit sizes (−0.25 ↔ +0.25 and −0.75 ↔ +0.75) × both directions. Configured style/fire seeds are distributed across cases, rather than multiplying their count. Both transition panels now report `transition_edited_voxels` and `transition_edit_accuracy` in the validation JSON/CSV. The latter checks destination occupancy and material only where the targets differ; at least 50% must match, alongside the usual source-shape, persistence, stability and recovery requirements. Identical rasterized targets have zero edited voxels and no edit-accuracy score: these check persistence, not remodeling. Older transition scores are not reused as the incumbent when resuming, because their acceptance rule was weaker.

The gene-transition `growth.gif` demonstrates a short → tall branching tree and `rollouts/transition.json` records the genomes and switch time. To try the resulting checkpoint, enable **Live remodel this mature organism** in the Tree Genome Lab, change a shape slider, apply, and advance growth. This trains one edit per rollout; simultaneous edits and rapid repeated slider changes still need separate evaluation. Pipeline tests do not establish trained remodeling quality. The `full` schedule still covers basics + variation; both live-transition curricula are separate choices.

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

Family training uses stratified low/high counterfactual pairs: seed, style, environment, fire masks, and damage are shared while exactly one gene changes. The loss combines balanced occupancy/material terms with soft Dice/IoU, distance-to-target, height, width, volume, centroid, and separate trunk, branch, and leaf Dice losses. Counterfactual error is normalized over voxels where the paired targets differ, so sparse gene effects are not diluted by world volume. The model uses one shared perception backbone with family-specific FiLM and output heads. Living masks, magnitude/range penalties, gradient clipping, and non-finite checks remain active.

Pairs teach the model what a gene changes: a short-branch and long-branch tree share the same background, so their output difference should match the target difference. This discourages growing one average tree while ignoring its controls, and prevents unrelated growth randomness from obscuring the comparison. Individual target and persistence losses still teach each tree's overall shape and stability. Neutral examples use identical genomes; the extra counterfactual loss is disabled during basic-family training and contributes zero for neutral pairs during variation.

Basic-family training grows each identical pair once, then copies the result into both pool entries. Batch 8 therefore grows four unique examples with the same loss weighting and gradients as eight duplicated examples. Paired pool entries remain available for checkpoint handoffs and later variation training, which continues to grow both members of each pair.

`gradient_accumulation: true` enables accumulation and `gradient_accumulation_steps` selects the number of physical batches per optimizer update. `iterations` continues to count optimizer updates, so a value of 8 uses roughly eight times the batch compute while retaining the memory footprint of one physical batch. Set `gradient_accumulation: false` to use one batch per update; the step count is then ignored.

The shipped tree-family preset uses one batch per optimizer update (`gradient_accumulation: false`, `gradient_accumulation_steps: 1`), a `0.0003` learning rate, gradient clipping at `1.0`, and the same stronger occupancy-range/magnitude penalties used by the stability stages. Its default 16³/batch-8 setting has an effective batch of 8. To restore the previous effective batch of 32, enable accumulation and set its step count to 4; that also restores four times the batch computation per optimizer update. Using fewer batches changes training statistics and may require different iteration counts for comparable quality. Validation runs every 500 optimizer updates to limit long-panel overhead.

`latest.pt` is the final optimizer state. `best.pt` is updated when the configured deterministic validation panel matches or improves its worst-case score, so an early zero-score tie cannot freeze the pipeline at its first validation window. A checkpoint is not stable merely because it is named `best.pt`; inspect its validation report and require `accepted: true`. Full tree presets validate for at least 256 steps and include recovery trials. Archive admission is stricter: the default minimum is 512 growth/persistence steps plus 128 recovery steps across fixed stochastic and environmental cases.

Variation validation covers every family with boundary, corner, random, interpolation, and mutation cases. The random/mutation counts and interpolation steps now apply **per family**; zero still disables that category. The default Phase 2 variation panel contains 104 trials instead of 32, so validation takes longer while checking all four families equally. Basic-family validation remains 32 trials.

Procedural tree targets use target version 4 and tree genomes use schema version 2. Wind response now depends on the actual wind vector supplied to the model, so identical inputs require identical targets. Version 3 checkpoint weights remain compatible; resuming them keeps weights and optimizer state but rebuilds the pool to remove targets generated with the old wind calculation. Saved validation results should be rerun with the corrected targets and expanded panel. Target versions before 3 and older genome schemas remain incompatible.

A “new variant” means a new valid genome/style-seed combination, not proof of a fundamentally new species. Mutation and interpolation stay inside the declared gene bounds, and interpolation is allowed only within one discrete family. A candidate outside the sampled training distribution can still fail; archive admission requires finite, bounded, connected, persistent, and regenerative validation rather than visual appeal alone.

## GPU guidance

`device: auto` selects CUDA when available, then Apple Metal (MPS), then CPU. The full presets use FP32 and a `16³` world; family uses batch 8, while the longer regeneration/environment horizons use batch 4 on an RTX 4050 Laptop GPU with 6 GB VRAM. Long 256–512-step validation runs under no-gradient inference.

Family rollouts prepare each example's genome modulation and output matrices once, then reuse them at every cellular step while preserving gradients. Growth and persistence losses share target-only calculations, material loss uses a fixed-shape background mask, and loss metrics transfer to the CPU together once per optimizer update. Apple Metal uses direct neighbor arithmetic for fixed 3D perception, with the same zero padding, channel order, and gradients as the convolution. CPU/CUDA retain cached convolution filters. Existing checkpoint parameter names and shapes remain compatible. These execution changes take effect in newly started processes; an already running training job keeps its loaded code and saved configuration.

Family sampling keeps bounded CPU caches of targets, distance maps, and environment fields keyed by the exact genome, environment, and grid size. Distance maps travel with their paired targets in the state pool; older checkpoints rebuild them when loaded. Pair sampling reuses structural metadata until those entries are replaced, and disabled damage skips damage ranking while retaining dead-state and routine reseeding.

Set **Hidden layers (neurons each)** in the dashboard to `32, 32`, or add this to a training preset:

```yaml
hidden_layers: [32, 32]
```

Each number specifies one hidden layer, with ReLU between layers; unequal widths such as `[64, 32]` also work. The list overrides `model_width`; omitting it retains the original single-layer model. `hidden_channels` remains the number of memory channels stored in each voxel. Genome modulation acts on the last shared hidden layer before the family output head. Checkpoints preserve the layer list, and training, the checkpoint viewer, evaluation, and ecology use it. Resuming or transferring specialist/family weights requires matching hidden layers: train a specialist with the new layout before handing it to Phase 2, or start a fresh model without a checkpoint. Existing single-layer checkpoints remain supported.

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
