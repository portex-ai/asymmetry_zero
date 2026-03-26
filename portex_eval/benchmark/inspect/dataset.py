import json
from mimetypes import guess_type
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample, json_dataset
from inspect_ai.model import (
    ChatMessageUser,
    ContentDocument,
    ContentImage,
    ContentText,
)


def dataset_generator(index_path: str) -> Any:
    data_path = Path(index_path)
    data_root = data_path.parent

    def content_from_reference(reference_path: Path) -> ContentDocument | ContentImage:
        mime_type = guess_type(reference_path.as_posix())[0]
        if not mime_type:
            raise ValueError(f"Unsupported reference type: {reference_path}")
        top_level = mime_type.split("/")[0]
        if top_level == "image":
            return ContentImage(image=str(reference_path))
        if top_level in {"application", "text"}:
            return ContentDocument(document=str(reference_path))
        raise ValueError(f"Unsupported mime type: {mime_type}")

    def record_to_sample(record: dict[str, Any]) -> Sample:
        reference_file = (record.get("reference_file") or "").strip()
        task_prompt = record.get("task_prompt") or record.get("prompt") or record.get("task")
        if not task_prompt:
            raise KeyError("task_prompt missing from task record")
        content: list[Any] = [ContentText(text=task_prompt)]
        if reference_file:
            reference_path = Path(reference_file)
            if not reference_path.is_absolute():
                reference_path = data_root / "refs" / reference_path
            content.append(content_from_reference(reference_path))

        return Sample(
            id=record.get("task_id"),
            input=[ChatMessageUser(content=content)],
            metadata={"reference_file": reference_file},
        )

    if data_path.suffix.lower() != ".jsonl":
        try:
            payload = json.loads(data_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None

        if isinstance(payload, dict) and isinstance(payload.get("prompts"), list):
            samples = [record_to_sample(record) for record in payload["prompts"]]
            return MemoryDataset(
                samples=samples,
                name=data_path.stem,
                location=data_path.as_posix(),
            )

        if isinstance(payload, list):
            samples = [record_to_sample(record) for record in payload]
            return MemoryDataset(
                samples=samples,
                name=data_path.stem,
                location=data_path.as_posix(),
            )

    return json_dataset(index_path, record_to_sample)
