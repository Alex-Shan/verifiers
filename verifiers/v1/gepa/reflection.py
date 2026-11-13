"""Bounded parallel teacher calls with prompt validation and response diagnostics."""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from gepa.strategies.instruction_proposal import InstructionProposalSignature

from verifiers.v1.configs.judge import JudgeConfig
from verifiers.v1.dialects.chat import message_to_wire, parse_message
from verifiers.v1.gepa.config import GEPAConfig
from verifiers.v1.judge import Judge
from verifiers.v1.utils.logging import get_logger

logger = get_logger(__name__)


def _normalize_prompt(prompt):
    """Accept GEPA chat wire dictionaries and native messages without changing strings."""
    if isinstance(prompt, str):
        return prompt
    messages = []
    for message in prompt:
        if isinstance(message, dict):
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError("Unsupported GEPA reflection message role")
            message = parse_message(message)
        messages.append(message)
    return messages


class ReflectionLM:
    """Expose GEPA's callable and batch_complete interfaces on one async path."""

    def __init__(self, judge: Judge, path: Path | None, max_concurrent: int = 4):
        self.cursor = 0
        self.judge = judge
        self.path = path
        self.max_concurrent = max_concurrent

    def __call__(self, prompt: Any) -> str:
        return self.batch_complete([prompt])[0]

    def batch_complete(self, messages_list: list[Any]) -> list[str]:
        return asyncio.run(self._batch_complete(messages_list))

    async def _batch_complete(self, prompts: list[Any]) -> list[str]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = await asyncio.gather(
            *(self._complete(prompt, semaphore) for prompt in prompts),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results

    async def _complete(self, prompt: Any, semaphore: asyncio.Semaphore) -> str:
        prompt_id = hashlib.sha256(
            json.dumps(
                prompt, ensure_ascii=False, sort_keys=True, default=message_to_wire
            ).encode()
        ).hexdigest()
        request_prompt = _normalize_prompt(prompt)
        call = self.cursor
        self.cursor += 1
        cache = (
            self.path.parent / "recovery" / f"teacher-{call}.json"
            if self.path
            else None
        )
        if cache and cache.exists():
            saved = json.loads(cache.read_text())
            if saved["prompt_id"] != prompt_id:
                raise ValueError("Recovery teacher identity mismatch")
            return saved["response"]
        for attempt in range(1, 3):
            async with semaphore:
                started = time.perf_counter()
                response = await self.judge.complete(request_prompt)
                error = None
                try:
                    instruction = InstructionProposalSignature.output_extractor(
                        response.text.strip()
                    )["new_instruction"]
                    if not instruction.strip():
                        error = "empty replacement prompt"
                except (KeyError, ValueError) as exc:
                    error = str(exc)
                if not response.text.strip():
                    error = "empty response content"
                if self.path is not None:
                    with self.path.open("a", encoding="utf-8") as file:
                        file.write(
                            json.dumps(
                                {
                                    "seconds": time.perf_counter() - started,
                                    "response_text": response.text,
                                    "usage": response.usage.model_dump(mode="json")
                                    if response.usage
                                    else None,
                                    "finish_reason": response.finish_reason,
                                    "prompt_id": prompt_id,
                                    "attempt": attempt,
                                    "valid_prompt": error is None,
                                    "validation_error": error,
                                }
                            )
                            + "\n"
                        )
                if error is None:
                    if cache:
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        temporary = cache.with_suffix(".tmp")
                        temporary.write_text(
                            json.dumps(
                                {"prompt_id": prompt_id, "response": response.text}
                            )
                        )
                        temporary.replace(cache)
                    return response.text
            if attempt < 2:
                logger.warning(
                    "Retrying reflection after invalid output (attempt 2/2) - %s",
                    error,
                )
        raise ValueError(f"Reflection returned no valid replacement prompt: {error}")


def build_reflection_lm(config: GEPAConfig) -> ReflectionLM:
    client = config.reflection_client or config.client
    judge = Judge(
        JudgeConfig(
            model=config.reflection_model or config.model,
            base_url=client.base_url,
            api_key_var=client.api_key_var,
            headers=client.headers,
            headers_from_env=client.headers_from_env,
            timeout=client.timeout,
        )
    )
    path = (
        config.output_dir / config.run.dir / "reflection_metrics.jsonl"
        if config.save_results
        else None
    )
    return ReflectionLM(judge, path)


EFFICIENCY_REFLECTION_TEMPLATE = """Write a compact replacement task system prompt.
Current task system prompt:
<curr_param>

Rollouts and feedback:
<side_info>

Optimize the stated weighted correctness and efficiency objective. The fixed harness
already documents tools; do not repeat it. Keep only generalizable instructions whose
benefit justifies their token cost. Do not accumulate examples or task-specific facts.
Allow additional text only when its expected correctness or execution savings justify
its cost. Before the replacement, explain observed success/failure differences and cost differences
between successful trajectories. Mark unavailable comparisons explicitly. Propose concrete
edits and explain their expected benefits and costs without claiming unmeasured gains.
Return the replacement instructions in exactly one fenced code block.
"""

TOKEN_REFLECTION_TEMPLATE = """Write a sufficient, compact replacement task system prompt.
Current task system prompt:
<curr_param>

Execution evidence:
<side_info>

Diagnose errors and unnecessary token expenditure from the observed trajectories.
Each task has one sample; do not invent same-task comparisons.
Explain concrete instruction edits and their expected correctness and token tradeoffs.
Favor appropriate numerical solvers over unnecessary large enumeration or repeated solving.
Preserve useful verification and allow non-tool solutions for suitable tasks.
Time is diagnostic only. Do not minimize tool use as an end in itself.
Do not repeat fixed harness tool documentation or accumulate task-specific examples.
Return exactly one fenced code block containing the replacement prompt.
"""
