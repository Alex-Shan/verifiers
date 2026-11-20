"""Run GEPA's teacher completions through the v1 Judge client."""

import asyncio

from verifiers.v1.configs.judge import JudgeConfig
from verifiers.v1.dialects.chat import parse_message
from verifiers.v1.gepa.config import GEPAConfig
from verifiers.v1.judge import Judge

ReflectionPrompt = str | list[dict]


class ReflectionLM:
    """Support GEPA's single and batch completion paths with one Judge."""

    def __init__(self, judge: Judge):
        self.judge = judge

    def __call__(self, prompt: ReflectionPrompt) -> str:
        return asyncio.run(self._complete(prompt))

    def batch_complete(self, prompts: list[ReflectionPrompt]) -> list[str]:
        return asyncio.run(self._complete_many(prompts))

    async def _complete_many(self, prompts: list[ReflectionPrompt]) -> list[str]:
        async with asyncio.TaskGroup() as group:
            calls = [group.create_task(self._complete(prompt)) for prompt in prompts]
        return [call.result() for call in calls]

    async def _complete(self, prompt: ReflectionPrompt) -> str:
        messages = (
            prompt
            if isinstance(prompt, str)
            else [parse_message(msg) for msg in prompt]
        )
        response = await self.judge.complete(messages)
        return response.text


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
    return ReflectionLM(judge)
