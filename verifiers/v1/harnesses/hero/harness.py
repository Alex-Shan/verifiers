import json
from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.bash.harness import BASH_SYSTEM_PROMPT
from verifiers.v1.harnesses.utils.launch import (
    CHAT_PROGRAM_SOURCE,
    CompactionConfig,
    compaction_args,
    launch_chat_program,
)
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace


def bash_tool_definition() -> dict:
    """Return Hero's bash-only tool definition."""
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a non-interactive bash command in a Linux shell and return its "
                "combined stdout and stderr as text. Use for: file operations (ls, cat, "
                "find, grep), running code/scripts (python3, node, gcc), package/tool "
                "inspection, git operations. Do NOT use for interactive, long-running "
                "or blocking commands (vim, nano, less, ssh, tail -f, npm start, python REPL). "
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The complete, self-contained bash command string to execute. "
                            "Must be valid bash syntax, non-interactive and require no user "
                            "input. Can use pipes, redirects and chaining with &&, ||, ;. "
                            "Provide concrete arguments, no placeholders. Example: "
                            '"ls -la /tmp && pwd" or "python3 -c \'print(1+1)\'"'
                        ),
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }


class HeroHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    compaction: CompactionConfig | None = CompactionConfig()
    """Use the shared chat program's default context compaction policy."""


class HeroHarness(Harness[HeroHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = False
    SUPPORTS_RESUME = True
    SUPPORTS_TOOL_INTERCEPTION = True
    NEEDS_CONTAINER = False

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(CHAT_PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        tool_interception_url: str | None = None,
    ) -> ProgramResult:
        if mcp_urls:
            raise ValueError(
                "hero only supports the bash tool; MCP tools are not supported"
            )
        system_prompt, prompt = self.resolve_prompt(data)
        system_prompt = "\n\n".join(p for p in (BASH_SYSTEM_PROMPT, system_prompt) if p)
        # Resume already includes the injected prompt when no task system prompt was set.
        if (
            data.system_prompt is None
            and prompt is not None
            and not isinstance(prompt, str)
            and any(
                message.role == "system" and message.content == BASH_SYSTEM_PROMPT
                for message in prompt
            )
        ):
            system_prompt = None
        args = [
            "--bash",
            f"--bash-tool-definition={json.dumps(bash_tool_definition())}",
            *compaction_args(self.config.compaction),
        ]
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        return await launch_chat_program(
            CHAT_PROGRAM_SOURCE,
            self.config,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            system_prompt,
            prompt,
            extra_args=args,
            activate=False,
        )
