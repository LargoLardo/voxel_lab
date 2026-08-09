---
type: "query"
date: "2026-08-07T07:27:54.672726+00:00"
question: "please show how many steps are occuring per second, as well as an easier ui with sliders to adjust configs hi"
contributor: "graphify"
outcome: "useful"
source_nodes: ["_job_view()", "write_live_preview()", "rollout()", "train()", "run_ecology.py", "evaluate_regeneration.py"]
---

# Q: please show how many steps are occuring per second, as well as an easier ui with sliders to adjust configs hi

## Answer

Expanded from original query via vocab: [ui.py, train(), rollout(), write_live_preview(), _job_view(), run_ecology.py, evaluate_regeneration.py]. Added measured steps_per_second to live preview metadata for training, regeneration, and ecology; surfaced it in the job card; added native range sliders that update the existing YAML config.

## Outcome

- Signal: useful

## Source Nodes

- _job_view()
- write_live_preview()
- rollout()
- train()
- run_ecology.py
- evaluate_regeneration.py