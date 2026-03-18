import pandas as pd
from rich import print
eval_df = pd.read_csv('portex-eval/eval_runs/simple_bundle/2026-03-18T17-32-52Z/reports/eval_level.csv')
task_df = pd.read_csv('portex-eval/eval_runs/simple_bundle/2026-03-18T17-32-52Z/reports/task_level.csv')
criteria_df = pd.read_csv('portex-eval/eval_runs/simple_bundle/2026-03-18T17-32-52Z/reports/criterion_level.csv')
judge_df = pd.read_csv('portex-eval/eval_runs/simple_bundle/2026-03-18T17-32-52Z/reports/judgement_level.csv')


print(eval_df.iloc[0].to_dict())
print(task_df.iloc[0].to_dict())
print(criteria_df.iloc[0].to_dict())
print(judge_df.iloc[2].to_dict())

judge_df.shape