"""HERO Python harness: a chat-completions loop with one persistent Python tool."""

import json
from pathlib import Path
from typing import Literal

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.hero_python.prompt import HERO_PYTHON_SYSTEM_PROMPT
from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace

# Keep the shared dependencies, streaming client, retries and output bounds while
# replacing only the executable chat loop.
PROGRAM_SOURCE = (
    CHAT_PROGRAM_SOURCE.rsplit('\nif __name__ == "__main__":', 1)[0]
    + "\n"
    + Path(__file__).with_name("program.py").read_text()
)
KERNEL_SOURCE = Path(__file__).with_name("kernel.py").read_bytes()
# Must match STOP_MARKER in program.py.
STOP_MARKER = "VF_HERO_PYTHON_STOP="
SCRATCH_DIR = "/tmp"


class HeroPythonHarnessConfig(HarnessConfig):
    dialect: Literal["chat_completions"] = "chat_completions"
    kernel_python: str = "python3"
    """Interpreter for the REPL kernel; its environment provides the solver packages."""


class HeroPythonHarness(Harness[HeroPythonHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_TOOL_INTERCEPTION = True
    NEEDS_CONTAINER = True
    SUPPORTS_RESUME = False
    SUPPORTS_MCP = False
    SUPPORTS_SKILLS = False

    async def setup(self, runtime: Runtime) -> None:
        kernel = self.config.kernel_python
        result = await runtime.run(
            ["sh", "-c", 'command -v "$1"', "sh", kernel], self.config.resolved_env
        )
        if result.exit_code:
            raise RuntimeError(f"hero_python kernel interpreter not found: {kernel}")
        # Warm the uv script cache once per runtime; launch reuses it for the path.
        await runtime.prepare_uv_script(
            CHAT_PROGRAM_SOURCE, self.config.resolved_env, activate=False
        )

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
            raise ValueError("hero_python does not support MCP")
        system, prompt = self.resolve_prompt(data)
        system = "\n\n".join(p for p in (HERO_PYTHON_SYSTEM_PROMPT, system) if p)
        script_path = f"{SCRATCH_DIR}/.vf-hero-python-{trace.id}.py"
        messages_path = f"{SCRATCH_DIR}/.vf-hero-python-{trace.id}.json"
        kernel_path = f"{SCRATCH_DIR}/.vf-hero-python-kernel-{trace.id}.py"
        # Secrets stay in argv, like the shared chat program, so the kernel and its
        # bash children do not inherit them through the environment.
        args = [
            f"--kernel-python={self.config.kernel_python}",
            f"--kernel-script={kernel_path}",
            f"--tool-timeout={self.config.tool_timeout}",
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
        ]
        if tool_interception_url:
            args.append(f"--tool-interception-url={tool_interception_url}")
        if system:
            args.append(f"--system-prompt={system}")
        if isinstance(prompt, str):
            args.append(f"--prompt={prompt}")
        elif prompt is not None:
            await runtime.write(
                messages_path,
                json.dumps([message_to_wire(message) for message in prompt]).encode(),
            )
            args.append(f"--initial-messages-file={messages_path}")
        interpreter, _ = await runtime.prepare_uv_script(
            CHAT_PROGRAM_SOURCE, self.config.resolved_env, activate=False
        )
        await runtime.write(script_path, PROGRAM_SOURCE.encode())
        await runtime.write(kernel_path, KERNEL_SOURCE)
        try:
            result = await runtime.run(
                [interpreter, script_path, *args], self.config.resolved_env
            )
        finally:
            await runtime.run(["rm", "-f", script_path, messages_path, kernel_path], {})
        for line in result.stderr.splitlines():
            if line.startswith(STOP_MARKER):
                # A distinct condition keeps these traces separable in analysis
                # without changing how they are scored.
                trace.stop(line.removeprefix(STOP_MARKER).strip())
        return result
