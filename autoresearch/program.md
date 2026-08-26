# MorphoVoxel autoresearch agent

You are running exactly one experiment in a frozen MorphoVoxel research campaign.

1. Read `configs/autoresearch_candidate.yaml` and the recent records named in the campaign context.
2. State one specific, falsifiable hypothesis about training quality or efficiency.
3. Edit only the paths listed under **Editable paths** in the campaign context. In the default config-only scope, edit only `configs/autoresearch_candidate.yaml`.
4. Do not edit targets, validation, metrics, benchmark seeds, the runner, tests, checkpoints, or normal dashboard runs. Do not commit, revert, delete, or rename files.
5. Run exactly one trial:

   `.venv\Scripts\python -m morphovoxel.autoresearch --trial --hypothesis "YOUR HYPOTHESIS"`

6. Read the returned record. Briefly report whether it was kept, discarded, crashed, timed out, or was rejected by the immutable-file guard, then stop.

The runner owns smoke tests, frozen overrides, validation, ranking, commits, and rollback. Never change evaluation code to improve a score. Prefer one small change per trial so causality remains legible.
