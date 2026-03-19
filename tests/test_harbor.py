"""Tests for Harbor-backed agent eval support."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from harbor.llms.base import LLMResponse

from portex_eval.benchmark.harbor.agent import PortexMultimodalAgent, PortexMultimodalChat
from portex_eval.benchmark.harbor.adapter import create_agent_eval_bundle
from portex_eval.benchmark.harbor.results import write_harbor_artifacts
from portex_eval.benchmark.harbor.run import (
    PORTEX_MULTIMODAL_AGENT_IMPORT_PATH,
    run_harbor_tasks,
)


def _write_bundle(bundle_dir: Path) -> None:
    refs_dir = bundle_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "diagram.txt").write_text("reference", encoding="utf-8")
    (bundle_dir / "tasks.json").write_text(
        json.dumps(
            {
                "version": 2,
                "prompts": [
                    {
                        "task_id": "task-1",
                        "task_prompt": "Use the reference file and answer the question.",
                        "reference_file": "diagram.txt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "answers.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "task-1",
                    "reference_file": "diagram.txt",
                    "tools": ["bash"],
                    "criteria": [
                        {
                            "id": "c1",
                            "name": "Exact answer",
                            "weight": 100,
                            "grader_type": "ExactMatch",
                            "semanticPrompt": "diagram",
                        }
                    ],
                    "passThreshold": 100,
                }
            ]
        ),
        encoding="utf-8",
    )


def test_create_agent_eval_bundle_generates_harbor_task_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "bundle"
        bundle_dir.mkdir()
        _write_bundle(bundle_dir)

        result = create_agent_eval_bundle(
            bundle_dir=str(bundle_dir),
            output_dir=str(Path(tmpdir) / "agent-tasks"),
        )

        task_dir = Path(result.datasets_dir) / "portex_task-1"
        assert result.task_count == 1
        assert (task_dir / "task.toml").is_file()
        assert (task_dir / "instruction.md").is_file()
        assert (task_dir / "tests" / "portex_grade.py").is_file()
        assert (task_dir / "tests" / "runtime" / "portex_eval" / "grading" / "core.py").is_file()
        assert (task_dir / "environment" / "refs" / "diagram.txt").is_file()

        task_config = json.loads((task_dir / "tests" / "task_config.json").read_text(encoding="utf-8"))
        task_toml = (task_dir / "task.toml").read_text(encoding="utf-8")
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
        assert task_config["task_id"] == "task-1"
        assert task_config["reference_file"] == "diagram.txt"
        assert task_config["judge_models"]
        assert 'OPENROUTER_API_KEY = "${OPENROUTER_API_KEY}"' in task_toml
        assert "PORTEX_JUDGE_MODELS" not in task_toml
        assert "PORTEX_JUDGE_CONFIGS" not in task_toml
        assert "OPENAI_API_KEY" not in task_toml
        assert "ANTHROPIC_API_KEY" not in task_toml
        assert "Reference file path: `/app/refs/diagram.txt`" in instruction


def test_run_harbor_tasks_builds_command_and_env() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        dataset_dir = task_root / "datasets" / "portex_task-1" / "tests"
        dataset_dir.mkdir(parents=True)
        (task_root / "datasets" / "portex_task-1" / "task.toml").write_text(
            '[verifier.env]\nOPENROUTER_API_KEY = "${OPENROUTER_API_KEY}"\nPORTEX_JUDGE_MODELS = "${PORTEX_JUDGE_MODELS}"\nPORTEX_JUDGE_CONFIGS = "${PORTEX_JUDGE_CONFIGS}"\n',
            encoding="utf-8",
        )
        (dataset_dir / "task_config.json").write_text("{}", encoding="utf-8")

        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch("subprocess.run") as mock_subprocess,
        ):
            result = run_harbor_tasks(
                task_root=str(task_root),
                judges=[{"provider": "openai", "model": "gpt-4o-mini"}],
                n_concurrent=2,
                env="local",
                extra_args=["--model", "demo-agent"],
            )

        cmd = mock_subprocess.call_args.args[0]
        env = mock_subprocess.call_args.kwargs["env"]
        assert cmd[:4] == [mock_subprocess.call_args.args[0][0], "-m", "harbor.cli.main", "run"]
        assert "--n-concurrent" in cmd
        assert "--env" in cmd
        assert env["PORTEX_JUDGE_MODELS"] == "openai:gpt-4o-mini"
        assert "PORTEX_JUDGE_CONFIGS" in env
        rewritten_toml = (task_root / "datasets" / "portex_task-1" / "task.toml").read_text(
            encoding="utf-8"
        )
        updated_task_config = json.loads((dataset_dir / "task_config.json").read_text(encoding="utf-8"))
        assert "PORTEX_JUDGE_MODELS" not in rewritten_toml
        assert "PORTEX_JUDGE_CONFIGS" not in rewritten_toml
        assert updated_task_config["judge_models"] == ["openai:gpt-4o-mini"]
        assert updated_task_config["judge_configs"][0]["provider"] == "openai"
        assert result.jobs_dir


def test_run_harbor_tasks_writes_jobs_to_separate_output_root() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir) / "tasks"
        output_root = Path(tmpdir) / "results"
        (task_root / "datasets").mkdir(parents=True)

        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch("subprocess.run"),
        ):
            result = run_harbor_tasks(
                task_root=str(task_root),
                output_root=str(output_root),
            )

        assert result.datasets_dir == str((task_root / "datasets").resolve())
        assert result.output_dir == str(output_root.resolve())
        assert Path(result.jobs_dir).parent == output_root.resolve() / "jobs"


def test_run_harbor_tasks_rewrites_portex_multimodal_agent_alias() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        (task_root / "datasets").mkdir()

        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch("subprocess.run") as mock_subprocess,
            patch(
                "portex_eval.benchmark.harbor.results.write_harbor_artifacts",
                return_value=(
                    ("eval.csv", "task.csv", "criterion.csv", "judgement.csv"),
                    type("RewardsPayload", (), {"task_ids": ["task-1"], "reward": [100.0]})(),
                    "rl_rewards.json",
                    "rl_training_data.json",
                ),
            ),
        ):
            run_harbor_tasks(
                task_root=str(task_root),
                extra_args=[
                    "--agent",
                    "portex-multimodal",
                    "--model",
                    "openrouter/google/gemini-3.1-pro-preview",
                ],
            )

        cmd = mock_subprocess.call_args.args[0]
        assert "--agent" not in cmd
        assert "--agent-import-path" in cmd
        assert PORTEX_MULTIMODAL_AGENT_IMPORT_PATH in cmd


def test_run_harbor_tasks_injects_model_info_for_known_vision_models() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        (task_root / "datasets").mkdir()

        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch("subprocess.run") as mock_subprocess,
            patch(
                "portex_eval.benchmark.harbor.results.write_harbor_artifacts",
                return_value=(
                    ("eval.csv", "task.csv", "criterion.csv", "judgement.csv"),
                    type("RewardsPayload", (), {"task_ids": ["task-1"], "reward": [100.0]})(),
                    "rl_rewards.json",
                    "rl_training_data.json",
                ),
            ),
        ):
            run_harbor_tasks(
                task_root=str(task_root),
                extra_args=["--model", "openrouter/google/gemini-3.1-pro-preview"],
            )

        cmd = mock_subprocess.call_args.args[0]
        ak_values = [cmd[idx + 1] for idx, arg in enumerate(cmd[:-1]) if arg == "--ak"]
        assert any(value.startswith("model_info=") for value in ak_values)


def test_run_harbor_tasks_materializes_judge_api_keys() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        (task_root / "datasets").mkdir()

        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch("subprocess.run") as mock_subprocess,
            patch(
                "portex_eval.benchmark.harbor.results.write_harbor_artifacts",
                return_value=(
                    ("eval.csv", "task.csv", "criterion.csv", "judgement.csv"),
                    type("RewardsPayload", (), {"task_ids": ["task-1"], "reward": [100.0]})(),
                    "rl_rewards.json",
                    "rl_training_data.json",
                ),
            ),
            patch.dict("os.environ", {"OPENAI_API_KEY": "openai-test-key"}, clear=False),
        ):
            run_harbor_tasks(
                task_root=str(task_root),
                judges=[{"provider": "openai", "model": "gpt-4o-mini"}],
            )

        env = mock_subprocess.call_args.kwargs["env"]
        judge_configs = json.loads(env["PORTEX_JUDGE_CONFIGS"])
        assert judge_configs[0]["provider"] == "openai"
        assert judge_configs[0]["model"] == "gpt-4o-mini"
        assert judge_configs[0]["api_key"] == "openai-test-key"
        assert "api_key_env" not in judge_configs[0]


def test_run_harbor_tasks_requires_harbor_install() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        (task_root / "datasets").mkdir()

        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ModuleNotFoundError, match="uv sync --group harbor"):
                run_harbor_tasks(task_root=str(task_root))


def test_write_harbor_artifacts_emits_reports_and_rl_files() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "agent-run"
        jobs_dir = output_dir / "jobs" / "run-1" / "task-1" / "logs" / "verifier"
        jobs_dir.mkdir(parents=True)
        detail = {
            "task_id": "task-1",
            "question": "What is in the file?",
            "submission": "Answer: diagram",
            "reference_file": "diagram.txt",
            "pass_threshold": 100,
            "total_score": 1.0,
            "total_score_raw": 100.0,
            "passed": True,
            "grade": "C",
            "judge_names": ["ExactMatch"],
            "criteria_results": [
                {
                    "criterion_id": "c1",
                    "name": "Exact answer",
                    "prompt": "diagram",
                    "semanticPrompt": "diagram",
                    "grader_type": "ExactMatch",
                    "weight": 100,
                    "grade": "C",
                    "passed": True,
                    "awarded": 100.0,
                    "judges": [
                        {
                            "model": "ExactMatch",
                            "grade": "C",
                            "passed": True,
                            "awarded": 100.0,
                            "explanation": "matched",
                        }
                    ],
                }
            ],
            "error": None,
        }
        (jobs_dir / "portex_detail.json").write_text(json.dumps(detail), encoding="utf-8")

        report_paths, rewards_payload, rewards_path, training_data_path = write_harbor_artifacts(
            jobs_dir=str(output_dir / "jobs"),
            output_dir=str(output_dir),
            run_id="run-1",
            datasets_dir=str(output_dir / "datasets"),
            agent_model="demo-agent",
            harbor_args=["--model", "demo-agent"],
        )

        assert Path(report_paths[0]).is_file()
        assert Path(report_paths[1]).is_file()
        assert Path(report_paths[2]).is_file()
        assert Path(report_paths[3]).is_file()
        assert Path(rewards_path).is_file()
        assert Path(training_data_path).is_file()
        assert rewards_payload.task_ids == ["task-1"]

        training_data = json.loads(Path(training_data_path).read_text(encoding="utf-8"))
        assert training_data["records"][0]["completion"] == "Answer: diagram"


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage = {"input_tokens": 12, "output_tokens": 5}
        self.raw = {
            "choices": [
                {
                    "message": {"content": text},
                }
            ]
        }


class _FakeProvider:
    provider_id = "openrouter"
    model_name = "google/gemini-3.1-pro-preview"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def agenerate(self, prompt: str, **kwargs: object) -> _FakeResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        return _FakeResponse("Answer: spotted cat")


class _FakeEnvironment:
    def __init__(self, task_config: dict[str, object], refs: dict[str, bytes]) -> None:
        self._task_config = task_config
        self._refs = refs
        self.exec_calls: list[str] = []
        self.uploads: dict[str, str] = {}

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        destination = Path(target_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source_path == "/tests/task_config.json":
            destination.write_text(json.dumps(self._task_config), encoding="utf-8")
            return
        destination.write_bytes(self._refs[source_path])

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.uploads[target_path] = Path(source_path).read_text(encoding="utf-8")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> object:
        del cwd, env, timeout_sec
        self.exec_calls.append(command)
        return object()


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def call(
        self,
        prompt: str,
        message_history: list[dict[str, object]] = [],
        logging_path: Path | None = None,
        **kwargs: object,
    ) -> LLMResponse:
        del logging_path, kwargs
        self.calls.append({"prompt": prompt, "message_history": message_history})
        return LLMResponse(
            content='{"analysis":"ok","plan":"done","commands":[],"task_complete":true}',
            usage=type(
                "Usage",
                (),
                {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "cache_tokens": 0,
                    "cost_usd": 0.0,
                },
            )(),
        )

    def get_model_context_limit(self) -> int:
        return 100000

    def get_model_output_limit(self) -> int | None:
        return 8192


def test_portex_multimodal_agent_builds_first_turn_image_message(tmp_path: Path) -> None:
    fake_provider = _FakeProvider()
    fake_environment = _FakeEnvironment(
        task_config={},
        refs={"/app/refs/image.png": b"\x89PNG\r\n\x1a\nfake"},
    )
    with patch("portex_eval.benchmark.harbor.agent.get_provider", return_value=fake_provider):
        agent = PortexMultimodalAgent(
            logs_dir=tmp_path / "logs",
            model_name="openrouter/google/gemini-3.1-pro-preview",
        )
        first_user_content, first_user_shadow, reference_meta = asyncio.run(
            agent._build_initial_user_message(
                environment=fake_environment,
                instruction="Solve it.\n\nReference file path: `/app/refs/image.png`\n",
                initial_prompt="Solve the task from the terminal.",
                temp_root=tmp_path,
            )
        )

    assert any(part.get("type") == "image" for part in first_user_content if isinstance(part, dict))
    assert "attached separately" in first_user_shadow
    assert reference_meta is not None
    assert reference_meta["mode"] == "image"


def test_portex_multimodal_agent_skips_missing_reference_placeholder(tmp_path: Path) -> None:
    fake_provider = _FakeProvider()
    fake_environment = _FakeEnvironment(task_config={}, refs={})
    with patch("portex_eval.benchmark.harbor.agent.get_provider", return_value=fake_provider):
        agent = PortexMultimodalAgent(
            logs_dir=tmp_path / "logs",
            model_name="openrouter/google/gemini-3.1-pro-preview",
        )
        first_user_content, first_user_shadow, reference_meta = asyncio.run(
            agent._build_initial_user_message(
                environment=fake_environment,
                instruction="Solve it.\n\nReference file path: `(none)`\n",
                initial_prompt="Solve the task from the terminal.",
                temp_root=tmp_path,
            )
        )

    assert first_user_content == [{"type": "text", "text": "Solve the task from the terminal."}]
    assert first_user_shadow == "Solve the task from the terminal."
    assert reference_meta is None


def test_portex_multimodal_chat_uses_multimodal_first_turn_and_stores_shadow() -> None:
    fake_llm = _FakeLLM()
    first_user_content = [
        {"type": "text", "text": "Initial prompt"},
        {"type": "image", "image": "/tmp/image.png", "detail": "high"},
    ]
    shadow = "Initial prompt\n\n[Reference image was attached separately: /app/refs/image.png]"
    chat = PortexMultimodalChat(
        fake_llm,
        first_user_content=first_user_content,
        first_user_shadow=shadow,
    )

    response = asyncio.run(chat.chat("ignored initial prompt"))

    assert response.content
    assert fake_llm.calls[0]["prompt"] == ""
    assert fake_llm.calls[0]["message_history"] == [
        {"role": "user", "content": first_user_content}
    ]
    assert chat.messages[0]["content"] == shadow
    assert chat.messages[1]["content"] == response.content
