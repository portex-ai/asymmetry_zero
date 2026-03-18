"""Minimal OpenRouter logprobs probe.

Run this script directly to verify whether a given OpenRouter model returns
completion logprobs for a simple prompt.

Example:
    uv run python examples/test_openrouter_logprobs.py \
        --model qwen/qwen3.5-9b \
        --prompt "What is the capital of France?" \
        --top-logprobs 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_payload(
    *,
    model: str,
    prompt: str,
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    logprobs: bool,
    top_logprobs: int | None,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": logprobs,
    }
    if top_logprobs is not None:
        payload["top_logprobs"] = top_logprobs
    return payload


def summarize_choice(choice: dict[str, Any]) -> None:
    message = choice.get("message", {})
    print("\n=== Completion ===")
    print(message.get("content", ""))

    print("\n=== Logprobs Status ===")
    if "logprobs" not in choice:
        print("`choices[0].logprobs` key is missing from the response.")
        return

    logprobs = choice.get("logprobs")
    if logprobs is None:
        print("`choices[0].logprobs` is present but null.")
        return

    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list):
        print("`choices[0].logprobs` is present but has no `content` list.")
        print(json.dumps(logprobs, indent=2))
        return

    print(f"Received logprobs for {len(content)} completion tokens.")
    for idx, token_info in enumerate(content[:10], start=1):
        token = token_info.get("token")
        token_logprob = token_info.get("logprob")
        top = token_info.get("top_logprobs")
        print(f"{idx:>2}. token={token!r} logprob={token_logprob}")
        if isinstance(top, list) and top:
            top_preview = [
                {"token": candidate.get("token"), "logprob": candidate.get("logprob")}
                for candidate in top[:5]
            ]
            print(f"    top_logprobs={json.dumps(top_preview)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe OpenRouter logprobs support.")
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument(
        "--system-prompt",
        default="Answer briefly, then end with a single final line: Answer: <text>",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument(
        "--no-logprobs",
        action="store_true",
        help="Disable logprobs to compare raw responses.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full raw response JSON.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required.")

    payload = build_payload(
        model=args.model,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        logprobs=not args.no_logprobs,
        top_logprobs=args.top_logprobs,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://portex.ai",
        "X-Title": "portex-eval-logprobs-test",
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved raw response to {args.output_json}")

    print("=== Request Settings ===")
    print(json.dumps(payload, indent=2))

    print("\n=== Response Summary ===")
    print(f"model={data.get('model')}")
    print(f"provider={data.get('provider')}")
    print(f"usage={json.dumps(data.get('usage', {}), indent=2)}")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SystemExit("No choices returned in response.")
    summarize_choice(choices[0])


if __name__ == "__main__":
    main()
