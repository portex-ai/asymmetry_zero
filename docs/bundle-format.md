# Bundle Format

This guide describes the structure and schema of Portex eval bundles.

## Overview

A Portex bundle is a directory containing evaluation tasks, reference answers, and optional supporting files:

```
mybenchmark/
├── tasks.json      # Task prompts with IDs
├── answers.json    # Reference answers and grading criteria
└── refs/           # Optional reference files (images, documents, etc.)
```

## tasks.json

The `tasks.json` file defines the prompts to be evaluated.

### Schema (version 2)

```json
{
  "version": 2,
  "prompts": [
    {
      "task_id": "unique-task-identifier",
      "task_prompt": "The question or instruction for the model",
      "reference_file": ""
    }
  ]
}
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | integer | Yes | Schema version (must be `2`) |
| `prompts` | array | Yes | List of task objects |
| `prompts[].task_id` | string | Yes | Unique identifier for the task |
| `prompts[].task_prompt` | string | Yes | The prompt text. Aliases: `prompt`, `task` |
| `prompts[].reference_file` | string | No | Path to reference file in `refs/` directory |

### Example

```json
{
  "version": 2,
  "prompts": [
    {
      "task_id": "geography-001",
      "task_prompt": "What is the capital of France?",
      "reference_file": ""
    },
    {
      "task_id": "vision-001",
      "task_prompt": "Describe what you see in this image.",
      "reference_file": "sample-image.png"
    }
  ]
}
```

## answers.json

The `answers.json` file contains reference answers and grading criteria.

### Schema

```json
[
  {
    "task_id": "unique-task-identifier",
    "answer": "The expected reference answer",
    "reference_file": "",
    "tools": [],
    "criteria": [],
    "passThreshold": 100
  }
]
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | string | Yes | Must match a task_id in tasks.json |
| `answer` | string | Yes | The reference answer text |
| `reference_file` | string | No | Path to reference file in `refs/` |
| `tools` | array | No | Tool definitions (reserved for future use) |
| `criteria` | array | No | Grading criteria for semantic evaluation |
| `passThreshold` | integer | No | Minimum score (0-100) to pass. Default: `100` |

### Criteria Schema

Criteria enable fine-grained semantic grading:

```json
{
  "id": "criterion-unique-id",
  "name": "criterion-name",
  "description": "Human-readable description of this criterion",
  "type": "semantic",
  "weight": 50,
  "rationale": "Why this criterion matters",
  "examples": [],
  "semanticPrompt": "Instruction for the judge to evaluate this criterion"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for the criterion |
| `name` | string | Yes | Short name for the criterion |
| `description` | string | Yes | Description of what this criterion evaluates |
| `type` | string | Yes | Criterion type (currently `semantic`) |
| `weight` | integer | Yes | Weight 0-100, criteria weights should sum to 100 |
| `rationale` | string | No | Explanation of why this criterion matters |
| `examples` | array | No | Example responses for calibration |
| `semanticPrompt` | string | Yes | Prompt given to judge for evaluation |

### Example with Criteria

```json
[
  {
    "task_id": "history-001",
    "answer": "World War II ended in 1945 with the surrender of Germany in May and Japan in September.",
    "reference_file": "",
    "tools": [],
    "criteria": [
      {
        "id": "history-001-c1",
        "name": "correct-year",
        "description": "States the correct year (1945).",
        "type": "semantic",
        "weight": 50,
        "rationale": "The year is the core factual element.",
        "examples": [],
        "semanticPrompt": "The response must state that World War II ended in 1945."
      },
      {
        "id": "history-001-c2",
        "name": "surrender-details",
        "description": "Mentions key surrender events.",
        "type": "semantic",
        "weight": 50,
        "rationale": "Additional context about how the war ended.",
        "examples": [],
        "semanticPrompt": "The response should mention the surrenders of Germany and/or Japan."
      }
    ],
    "passThreshold": 75
  }
]
```

## refs/ Directory

The `refs/` directory contains reference files used by tasks:

- Images (PNG, JPEG, WebP, GIF)
- Documents (PDF, text files)
- Any supporting assets

Files are referenced by their path relative to `refs/`:

```json
{
  "task_id": "vision-001",
  "task_prompt": "Describe this flag.",
  "reference_file": "flags/spain.webp"
}
```

Corresponding file location: `mybenchmark/refs/flags/spain.webp`

## BYOB Format

For quick prototyping, use the simplified "Bring Your Own Benchmark" JSON format with `create_benchmark()`:

```json
[
  {
    "task": "What is the capital of France?",
    "answer": "Paris",
    "reference_file": ""
  }
]
```

This is converted to the full bundle format with auto-generated task IDs and default grading settings.

### BYOB Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task` | string | Yes | The task prompt |
| `answer` | string | Yes | The reference answer |
| `reference_file` | string | No | Path to a reference file |

### Converting BYOB to Bundle

```python
from portex_eval import create_benchmark

benchmark = create_benchmark("./mybench.json")
print(f"Bundle created at: {benchmark.path}")
print(f"Task count: {benchmark.task_count}")
```

The output directory is created adjacent to the input file with the same name (minus extension):
- Input: `./mybench.json`
- Output: `./mybench/tasks.json`, `./mybench/answers.json`, `./mybench/refs/`

## Validation

Bundles are validated when passed to `eval()`:

1. `tasks.json` must have `version: 2`
2. Each task must have a non-empty `task_id` and `task_prompt`
3. `answers.json` must be a list
4. Each answer must reference a valid `task_id` from `tasks.json`
5. Reference files must exist in the `refs/` directory

Validation errors raise `PortexEvalError` with specific messages.
