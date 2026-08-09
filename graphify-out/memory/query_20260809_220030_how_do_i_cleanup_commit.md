---
type: "query"
date: "2026-08-09T22:00:30.599505+00:00"
question: "how do i cleanup commit"
contributor: "graphify"
outcome: "dead_end"
---

# Q: how do i cleanup commit

## Answer

Expanded from original query via vocab: [files, run, runs, test, tests]. The graph did not contain Git-index cleanup guidance, so current Git state was inspected directly. Keep graphify-out/ in .gitignore, add .tmp-*/ and .verification-runs/, run git rm -r --cached with the seven named .tmp directories and .verification-runs, stage only .gitignore, inspect git diff --cached --name-status, commit with Remove generated test artifacts, and push origin main. Do not use git add . because README, trainer, UI, and test files have unrelated unstaged modifications.

## Outcome

- Signal: dead_end