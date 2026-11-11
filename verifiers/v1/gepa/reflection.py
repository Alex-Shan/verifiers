"""The reflection (teacher) LM: GEPA's proposer calls this synchronously to turn a prompt into
text. Backed by a vf `Judge` — `Judge.complete` owns the model client (same key / header /
Prime-billing resolution rollouts use), so we never build one here. GEPA calls it from the
worker thread running `optimize()`, so each call drives the async `complete` on a private loop.
"""

import asyncio
import json
import time
from collections.abc import Callable

from verifiers.v1.configs.judge import JudgeConfig
from verifiers.v1.gepa.config import GEPAConfig
from verifiers.v1.judge import Judge


def build_reflection_lm(config: GEPAConfig) -> Callable[[str], str]:
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

    def reflection_lm(prompt: str) -> str:
        started = time.perf_counter()
        response = asyncio.run(judge.complete(prompt))
        if config.save_results:
            path = config.output_dir / config.run.dir / "reflection_metrics.jsonl"
            with path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "seconds": time.perf_counter() - started,
                            "response_text": response.text,
                            "usage": response.usage.model_dump(mode="json")
                            if response.usage
                            else None,
                        }
                    )
                    + "\n"
                )
        return response.text

    return reflection_lm


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
