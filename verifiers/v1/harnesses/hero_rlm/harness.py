"""Prime Agent execution with minimal HERO Python guidance."""

from verifiers.v1.acp import ACPConfig
from verifiers.v1.clients import ModelContext
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.hero_rlm.prompt import HERO_RLM_SYSTEM_PROMPT
from verifiers.v1.harnesses.prime_agent.harness import (
    PrimeAgentHarness,
    PrimeAgentHarnessConfig,
)
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


class HeroRlmHarnessConfig(PrimeAgentHarnessConfig):
    """Use the same pinned release and execution settings as Prime Agent."""


class HeroRlmHarness(PrimeAgentHarness, Harness[HeroRlmHarnessConfig]):
    SUPPORTS_MCP = False
    SUPPORTS_SKILLS = False

    async def prepare_acp(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ACPConfig:
        if mcp_urls:
            raise ValueError("hero_rlm supports the Python kernel only, without MCP")
        config = await super().prepare_acp(
            ctx, trace, runtime, endpoint, secret, mcp_urls, data
        )
        config.command.extend(
            ["--system-prompt", HERO_RLM_SYSTEM_PROMPT, "--no-skills"]
        )
        return config
