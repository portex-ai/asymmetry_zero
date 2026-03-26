# Bundle Format

This guide describes the bundle format consumed by `eval()` and produced by `create_benchmark()`.

## Directory Layout

```text
my_bundle/
├── tasks.json
├── answers.json
└── refs/
```

`refs/` is optional unless a task uses `reference_file`.

## `tasks.json`

Recommended canonical shape:

```json
{
  "version": 2,
  "prompts": [
    {
      "task_id": "capital-001",
      "task_prompt": "What is the capital of France?",
      "reference_file": ""
    }
  ]
}
```

Supported task fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `task_id` | Yes | Unique task id |
| `task_prompt` | Yes | Prompt text |
| `reference_file` | No | File path relative to `refs/` |
| `metadata` | No | Extra metadata preserved for Harbor task generation |
| `environment` | No | Harbor-oriented environment settings |

The loader also accepts legacy plain-list `tasks.json` payloads. Within each task record it accepts `task_prompt`, `prompt`, or `task`, but new bundles should write `task_prompt`.

## `answers.json`

`answers.json` is always a list:

```json
[
  {
    "task_id": "capital-001",
    "reference_file": "",
    "tools": [],
    "criteria": [
      {
        "id": "capital-exact",
        "name": "Exact capital",
        "weight": 100,
        "grader_type": "ExactMatch",
        "semanticPrompt": "Paris"
      }
    ],
    "passThreshold": 100
  }
]
```

Supported fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `task_id` | Yes | Must match a task in `tasks.json` |
| `reference_file` | No | Optional mirror of the task's reference file |
| `tools` | No | Reserved tool list for future workflows |
| `criteria` | Yes | Non-empty list of grading criteria |
| `passThreshold` | No | Minimum passing score, default `100` |

## Criteria

Each criterion must include:

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | Yes | Unique criterion id |
| `weight` | Yes | Numeric weight |
| `grader_type` | Yes | `ExactMatch` or `llm-judge` |
| `semanticPrompt` | Recommended | Judge instruction or expected exact answer |

The validator accepts `semanticPrompt`, `description`, or `name` as the prompt-like field, but `semanticPrompt` is the clearest and should be present in public bundles.

Example mixed-criterion answer:

```json
[
  {
    "task_id": "history-001",
    "criteria": [
      {
        "id": "year",
        "name": "Correct year",
        "weight": 50,
        "grader_type": "ExactMatch",
        "semanticPrompt": "1945"
      },
      {
        "id": "context",
        "name": "Correct context",
        "weight": 50,
        "grader_type": "llm-judge",
        "semanticPrompt": "The response should mention the surrender of Germany or Japan."
      }
    ],
    "passThreshold": 75
  }
]
```

## Reference Files

Reference files live under `refs/` and are referenced relative to that directory:

```json
{
  "task_id": "vision-001",
  "task_prompt": "Count the books in the image.",
  "reference_file": "images/math-001.webp"
}
```

That maps to `my_bundle/refs/images/math-001.webp`.

## BYOB Input Format

`create_benchmark()` accepts a simpler input format:

```json
[
  {
    "task": "What is the capital of France?",
    "criteria": [
      {
        "id": "capital-exact",
        "name": "Exact capital",
        "weight": 100,
        "grader_type": "ExactMatch",
        "semanticPrompt": "Paris"
      }
    ],
    "reference_file": ""
  }
]
```

Supported BYOB fields:

| Field | Required | Meaning |
| --- | --- | --- |
| `task` | Yes | Prompt text |
| `criteria` | Yes | Non-empty criteria list |
| `reference_file` | No | Path resolved relative to the BYOB JSON file |
| `tools` | No | Optional tool list copied into `answers.json` |
| `passThreshold` | No | Minimum passing score, default `100` |

`create_benchmark()` writes a sibling bundle directory named like `<stem>_<random>/`.

## Validation Rules

Before running an eval, `portex-eval` checks that:

1. `tasks.json` exists and contains task ids plus prompt text.
2. `answers.json` exists and is a list.
3. Every answer record points at a known `task_id`.
4. Every answer record has at least one criterion.
5. Every criterion has an `id`, numeric `weight`, and valid `grader_type`.
6. Each criterion has at least one prompt-like field (`semanticPrompt`, `description`, or `name`).
7. Any referenced files exist.
