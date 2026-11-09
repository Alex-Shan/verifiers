"""Container-side chat program backed by a minimal persistent Python REPL kernel."""

import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path

import certifi
import httpx
from openai import APIStatusError, AsyncOpenAI, DefaultAsyncHttpxClient

from verifiers.v1.harnesses.utils.compaction import (
    Compactor,
    bound_tool_message,
    is_context_overflow,
)
from verifiers.v1.harnesses.utils.core import gate_tool_call

# Must match STOP_MARKER in harness.py.
STOP_MARKER = "VF_HERO_PYTHON_STOP="
RESTARTED_NOTE = (
    "\n[The Python kernel was restarted; all previously defined variables, imports, "
    "and functions are lost.]"
)

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python_repl",
        "description": "Execute Python in a persistent kernel; use bash(cmd) for shell commands.",
        "parameters": {
            "type": "object",
            "required": ["code"],
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."}
            },
        },
        "strict": False,
    },
}


class KernelDied(RuntimeError):
    """The kernel process exited or stopped responding."""


class Kernel:
    """Host side of kernel.py: one JSON line per request and per reply."""

    def __init__(self, python, script, timeout):
        self.python = python
        self.script = script
        self.timeout = timeout
        self.process = None

    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            self.python,
            "-u",
            self.script,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            limit=16 * 1024 * 1024,
        )

    async def restart(self):
        await self.close()
        await self.start()

    async def execute(self, code):
        process = self.process
        try:
            process.stdin.write((json.dumps({"code": code}) + "\n").encode())
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as error:
            raise KernelDied("exited") from error
        reply = asyncio.create_task(process.stdout.readline())
        try:
            raw = await asyncio.wait_for(asyncio.shield(reply), self.timeout)
            note = ""
        except TimeoutError:
            process.send_signal(signal.SIGINT)
            try:
                raw = await asyncio.wait_for(asyncio.shield(reply), 5)
            except TimeoutError:
                reply.cancel()
                raise KernelDied("interrupt") from None
            note = "TimeoutError: Python execution exceeded the tool timeout and was interrupted.\n"
        if not raw:
            raise KernelDied("exited")
        return note + json.loads(raw)["output"]

    async def run_tool(self, code):
        """Execute model code; a dead or wedged kernel is restarted, not fatal."""
        try:
            return await self.execute(code)
        except KernelDied as error:
            await self.restart()
            reason = (
                "TimeoutError: Python execution exceeded the tool timeout and could not be interrupted."
                if str(error) == "interrupt"
                else "Error: the Python kernel exited while running this code."
            )
            return reason + RESTARTED_NOTE

    async def close(self):
        if self.process is None:
            return
        process = self.process
        self.process = None
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()


def report_stop(condition):
    print(f"{STOP_MARKER}{condition}", file=sys.stderr, flush=True)


async def python_chat(args, kernel, client, tool_client, messages):
    compactor = Compactor(client, args.model, [PYTHON_TOOL], False, None)
    compactor.note_good(messages)
    while True:
        try:
            completion, messages = await compactor.complete(messages)
        except APIStatusError as error:
            if is_context_overflow(error):
                report_stop("context_overflow")
                return
            raise
        message = completion.choices[0].message
        messages.append(message.model_dump(exclude_none=True))
        if not message.tool_calls:
            return
        for call in message.tool_calls:
            if args.tool_interception_url:
                decision = await gate_tool_call(
                    tool_client, args.tool_interception_url, args.api_key, call
                )
                if decision["action"] == "deny":
                    messages.append(bound_tool_message(decision["message"]))
                    continue
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as error:
                content = f"error: invalid JSON in tool arguments ({error})"
            else:
                if call.function.name != "python_repl":
                    content = f"error: unknown tool {call.function.name!r}"
                elif not isinstance(arguments, dict) or not isinstance(
                    arguments.get("code"), str
                ):
                    content = "error: python_repl requires an object with a code string"
                else:
                    content = await kernel.run_tool(arguments["code"])
            messages.append(
                bound_tool_message(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function.name,
                        "content": content,
                    }
                )
            )


async def main():
    parser = argparse.ArgumentParser()
    for name in ("base-url", "api-key", "model", "kernel-python", "kernel-script"):
        parser.add_argument(f"--{name}", required=True)
    for name in (
        "system-prompt",
        "prompt",
        "initial-messages-file",
        "tool-interception-url",
    ):
        parser.add_argument(f"--{name}", default="")
    parser.add_argument("--tool-timeout", type=float, default=600)
    args = parser.parse_args()
    task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    messages = (
        [{"role": "system", "content": args.system_prompt}]
        if args.system_prompt
        else []
    )
    if args.initial_messages_file:
        path = Path(args.initial_messages_file)
        messages.extend(json.loads(path.read_bytes()))
        path.unlink()
    elif args.prompt:
        messages.append({"role": "user", "content": args.prompt})
    kernel = Kernel(args.kernel_python, args.kernel_script, args.tool_timeout)
    async with (
        AsyncOpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=httpx.Timeout(None, connect=5.0),
            http_client=DefaultAsyncHttpxClient(verify=certifi.where()),
        ) as client,
        httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0)) as tool_client,
    ):
        try:
            await kernel.start()
            await python_chat(args, kernel, client, tool_client, messages)
        finally:
            await kernel.close()


if __name__ == "__main__":
    asyncio.run(main())
