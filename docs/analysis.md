# Analysis

This guide shows a minimal analysis workflow on top of the generated artifacts.

## Load Standard Artifacts

```python
import json
from pathlib import Path

from portex_eval import reports

eval_df = reports.load(results.reports.eval_level)
task_df = reports.load(results.reports.task_level)
criterion_df = reports.load(results.reports.criterion_level)
judgement_df = reports.load(results.reports.judgement_level)

rewards = json.loads(Path(results.rewards_path).read_text(encoding="utf-8"))
training_data = json.loads(Path(results.training_data_path).read_text(encoding="utf-8"))
```

## Basic Score Checks

```python
print(task_df["score"].describe())
print(task_df.sort_values("score").head(10)[["task_id", "score"]])
```

## Failed Tasks

```python
failed = task_df[task_df["score"] < task_df["PassThreshold"]]
print(failed[["task_id", "score", "PassThreshold"]])
```

If your downstream tables expose a different pass-threshold column casing, inspect `task_df.columns` first and adjust accordingly.

## Criterion-Level Breakdowns

```python
criterion_avg = criterion_df.groupby("criterion_name")["criteria_awarded"].mean()
print(criterion_avg.sort_values())
```

If you want per-task drilldowns:

```python
task_id = task_df.sort_values("score").iloc[0]["task_id"]
print(criterion_df[criterion_df["task_id"] == task_id])
```

## Judge-Level Inspection

```python
task_id = task_df.sort_values("score").iloc[0]["task_id"]
task_judgements = judgement_df[judgement_df["task_id"] == task_id]
print(task_judgements[["judge_name", "judge_grade", "judge_reasoning"]])
```

## Inspect `.eval` Logs

```python
from inspect_ai import read_eval_log
from inspect_ai.analysis import samples_df

log = read_eval_log(results.logs[0], header_only=False)
samples = samples_df(results.logs)

print(samples.head())
print(log.samples[0].id)
```

## Reward and Training Data Artifacts

The reward JSON is usually the easiest bridge into downstream training:

```python
reward_by_task = dict(zip(rewards["task_ids"], rewards["reward"]))
print(reward_by_task)
```

The training-data JSON already contains prompt/completion/reward tuples:

```python
records = training_data["records"]
print(records[0]["prompt_text"])
print(records[0]["completion"])
print(records[0]["reward"])
```
