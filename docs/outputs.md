# Outputs

This guide describes the artifacts written by `portex-eval`.

## Standard Eval Layout

A standard `eval()` run writes:

```text
<results.output_dir>/
├── logs/
│   └── *.eval
├── manifest.json
├── reports/
│   ├── eval_level.csv
│   ├── task_level.csv
│   ├── criterion_level.csv
│   └── judgement_level.csv
├── rl_rewards.json
└── rl_training_data.json
```

## Artifact Summary

| Artifact | Meaning |
| --- | --- |
| `logs/*.eval` | Inspect evaluation logs |
| `manifest.json` | Run metadata, bundle metadata, model specs, and selected log path |
| `reports/eval_level.csv` | Run-level summary metrics |
| `reports/task_level.csv` | Per-task scores and task metadata |
| `reports/criterion_level.csv` | Per-criterion grading results |
| `reports/judgement_level.csv` | Individual judge outputs |
| `rl_rewards.json` | Reward arrays for post-training pipelines |
| `rl_training_data.json` | Structured prompt/completion/reward records |

## Reward JSON

`rl_rewards.json` is a compact JSON payload:

```json
{
  "task_ids": ["task-1", "task-2"],
  "reward": [100.0, 87.5]
}
```

Programmatic use:

```python
import json
from pathlib import Path

payload = json.loads(Path(results.rewards_path).read_text(encoding="utf-8"))
print(payload["task_ids"])
print(payload["reward"])
```

The same payload is also exposed as `results.rewards`.

## Training Data JSON

`rl_training_data.json` is the richer artifact intended for post-training workflows.

Top-level structure:

```json
{
  "format": "portex-rl-training-data",
  "version": 1,
  "source": {
    "eval_log": "/abs/path/to/log.eval"
  },
  "records": []
}
```

Each record includes:

- `task_id`
- `prompt_messages`
- `prompt_text`
- `completion`
- `reward`
- `reference_file`
- `completion_logprobs` when logprobs were requested

## Loading CSV Reports

```python
from portex_eval import reports

eval_df = reports.load(results.reports.eval_level)
task_df = reports.load(results.reports.task_level)
criterion_df = reports.load(results.reports.criterion_level)
judgement_df = reports.load(results.reports.judgement_level)
```

## Inspect Logs

The `.eval` files are standard Inspect logs:

```python
from inspect_ai.analysis import eval_df, samples_df

evals = eval_df(results.logs)
samples = samples_df(results.logs)
```

## Harbor Outputs

Harbor-backed runs expose their important locations through `AgentEvalResults`:

| Field | Meaning |
| --- | --- |
| `output_dir` | Harbor results root |
| `datasets_dir` | Harbor datasets directory |
| `jobs_dir` | Harbor jobs directory for the run |
| `rewards_path` | Path to Harbor reward JSON |
| `training_data_path` | Path to Harbor training-data JSON |

When `output_dir` is not provided, `agent_eval()` uses the task root as the Harbor results root and writes the jobs directory under `jobs/<run_id>/`.
