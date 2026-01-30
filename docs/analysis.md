# Analysis

This guide covers analyzing evaluation results from `portex-eval`.

## Loading Results

After running an evaluation, load results using the `reports` module:

```python
from portex_eval import eval, reports

results = eval(
    path="./mybenchmark",
    judges=["openrouter:openai/gpt-4o", "openrouter:anthropic/claude-3.5-sonnet"],
    candidates=["openrouter:meta-llama/llama-3.3-70b-instruct"],
)

# Load all report DataFrames
eval_df = reports.load(results.reports.eval_level)
task_df = reports.load(results.reports.task_level)
criterion_df = reports.load(results.reports.criterion_level)
judgement_df = reports.load(results.reports.judgement_level)
```

## Basic Analysis

### Overall Performance

```python
# Mean score across all tasks
mean_score = task_df["score"].mean()
print(f"Mean score: {mean_score:.1f}")

# Pass rate
pass_rate = task_df["passed"].mean() * 100
print(f"Pass rate: {pass_rate:.1f}%")

# Score distribution
print(task_df["score"].describe())
```

### Identifying Problem Areas

```python
# Tasks that failed
failed_tasks = task_df[task_df["passed"] == False]
print(f"Failed tasks: {len(failed_tasks)}")

# Lowest scoring tasks
lowest = task_df.nsmallest(10, "score")[["task_id", "score"]]
print("Lowest scoring tasks:")
print(lowest)

# Perfect scores
perfect = task_df[task_df["score"] == 100.0]
print(f"Perfect scores: {len(perfect)}")
```

### Criterion-Level Analysis

```python
# Average score by criterion
criterion_avg = criterion_df.groupby("criterion_name")["score"].mean()
print("Average score by criterion:")
print(criterion_avg.sort_values())

# Which criteria are hardest?
hardest = criterion_df.groupby("criterion_name")["score"].mean().nsmallest(5)
print("Hardest criteria:")
print(hardest)
```

### Judge Agreement

```python
# Check judge consistency
judge_scores = judgement_df.pivot_table(
    index="task_id",
    columns="judge_model",
    values="verdict",
    aggfunc="first"
)
print("Judge verdicts by task:")
print(judge_scores.head(10))

# Agreement rate
# (Requires custom logic based on verdict format)
```

## Inspect AI Integration

For deeper analysis, use Inspect AI's built-in tools:

```python
from inspect_ai.analysis import eval_df, samples_df

# Load evaluation data
evals = eval_df(results.logs)
samples = samples_df(results.logs)

# Explore samples
print(samples.columns)
print(samples.head())

# Filter by score
low_scores = samples[samples["score"] < 50]
```

### Viewing Individual Samples

```python
from inspect_ai import read_eval_log

log = read_eval_log(results.logs[0])

for sample in log.samples[:5]:
    print(f"Task: {sample.id}")
    print(f"Prompt: {sample.input[:100]}...")
    print(f"Response: {sample.output[:100]}...")
    print(f"Score: {sample.score}")
    print("---")
```

## Visualization

### Score Distribution

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(task_df["score"], bins=20, edgecolor="black")
plt.xlabel("Score")
plt.ylabel("Count")
plt.title("Score Distribution")
plt.axvline(task_df["score"].mean(), color="red", linestyle="--", label="Mean")
plt.legend()
plt.savefig("score_distribution.png")
```

### Pass/Fail Breakdown

```python
pass_counts = task_df["passed"].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(pass_counts, labels=["Pass", "Fail"], autopct="%1.1f%%")
plt.title("Pass/Fail Distribution")
plt.savefig("pass_fail.png")
```

### Criterion Comparison

```python
criterion_avg = criterion_df.groupby("criterion_name")["score"].mean().sort_values()

plt.figure(figsize=(12, 6))
criterion_avg.plot(kind="barh")
plt.xlabel("Average Score")
plt.ylabel("Criterion")
plt.title("Average Score by Criterion")
plt.tight_layout()
plt.savefig("criteria_comparison.png")
```

## Comparing Runs

Compare results across multiple evaluation runs:

```python
import pandas as pd

# Load multiple runs
run1_tasks = reports.load("./eval_runs/run1/reports/task_level.csv")
run2_tasks = reports.load("./eval_runs/run2/reports/task_level.csv")

# Add run identifiers
run1_tasks["run"] = "run1"
run2_tasks["run"] = "run2"

# Combine
combined = pd.concat([run1_tasks, run2_tasks])

# Compare means
comparison = combined.groupby("run")["score"].agg(["mean", "std", "min", "max"])
print(comparison)

# Task-level comparison
merged = run1_tasks.merge(
    run2_tasks,
    on="task_id",
    suffixes=("_run1", "_run2")
)
merged["delta"] = merged["score_run2"] - merged["score_run1"]
print(merged[["task_id", "score_run1", "score_run2", "delta"]].sort_values("delta"))
```

## Exporting Results

### Export to Excel

```python
with pd.ExcelWriter("evaluation_results.xlsx") as writer:
    task_df.to_excel(writer, sheet_name="Tasks", index=False)
    criterion_df.to_excel(writer, sheet_name="Criteria", index=False)
    judgement_df.to_excel(writer, sheet_name="Judgements", index=False)
```

### Export Summary

```python
summary = {
    "total_tasks": len(task_df),
    "mean_score": task_df["score"].mean(),
    "median_score": task_df["score"].median(),
    "std_score": task_df["score"].std(),
    "pass_rate": task_df["passed"].mean(),
    "min_score": task_df["score"].min(),
    "max_score": task_df["score"].max(),
}

import json
with open("summary.json", "w") as f:
    json.dump(summary, f, indent=2)
```

## RL Pipeline Integration

For RL training, use the rewards file directly:

```python
def load_normalized_rewards(path: str) -> dict[str, float]:
    """Load rewards normalized to 0-1 range."""
    rewards = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                task_id, score = parts
                rewards[task_id] = float(score) / 100.0
    return rewards

rewards = load_normalized_rewards(results.rewards)

# Use in training
for batch in training_data:
    batch_rewards = [rewards.get(sample.id, 0.0) for sample in batch]
    # Apply to RL loss
```

## Debugging Failed Tasks

When tasks fail, investigate the judge reasoning:

```python
# Find a failed task
failed = task_df[task_df["passed"] == False].iloc[0]
task_id = failed["task_id"]

# Get all judgements for this task
task_judgements = judgement_df[judgement_df["task_id"] == task_id]

print(f"Task: {task_id}")
print(f"Score: {failed['score']}")
print("\nJudge verdicts:")
for _, j in task_judgements.iterrows():
    print(f"  {j['judge_model']}: {j['verdict']}")
    print(f"    Rationale: {j['rationale'][:200]}...")
```

This helps identify:
- Ambiguous task prompts
- Too-strict grading criteria
- Model capability gaps
- Judge disagreements
