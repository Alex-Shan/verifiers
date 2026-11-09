"""Minimal persistent Python REPL kernel for the HERO Python harness.

Protocol: one JSON object per line. The host sends {"code": str}; the kernel replies
{"output": str} after the cell finishes. The host interrupts a running cell with
SIGINT. Cell output (including subprocesses) is captured at the file-descriptor level
so it can never corrupt the protocol stream.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import traceback

HEAD = 32768
TAIL = 32768


def bash(command, timeout=None):
    """Run a shell command and return its combined stdout and stderr."""
    result = subprocess.run(
        command,
        shell=True,
        executable="/bin/bash",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        timeout=timeout,
        check=False,
    )
    output = result.stdout
    if result.returncode:
        output += f"\n[exit code {result.returncode}]"
    return output


def run_cell(code, namespace):
    try:
        tree = ast.parse(code, "<cell>", "exec")
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, "<cell>", "exec"), namespace)  # noqa: S102
        if last is not None:
            value = eval(compile(last, "<cell>", "eval"), namespace)
            if value is not None:
                print(repr(value))
    except SystemExit as error:
        print(f"SystemExit: {error.code}", file=sys.stderr)
    except Exception as error:  # noqa: BLE001 - report errors raised by user cells
        # Drop the kernel's own frame from the traceback.
        tb = error.__traceback__.tb_next if error.__traceback__ else None
        traceback.print_exception(type(error), error, tb)


def read_bounded(path):
    size = os.path.getsize(path)
    with open(path, "rb") as file:
        if size <= HEAD + TAIL:
            return file.read().decode("utf-8", "replace")
        head = file.read(HEAD).decode("utf-8", "ignore")
        file.seek(size - TAIL)
        tail = file.read().decode("utf-8", "ignore")
    return f"{head}\n[... {size - HEAD - TAIL} bytes truncated ...]\n{tail}"


def main():
    # Private copies of the protocol pipes: cell code (exit(), input(), subprocesses)
    # sees /dev/null on fds 0-2 and cannot consume or corrupt the protocol.
    requests = os.fdopen(os.dup(0), "r")
    protocol = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    namespace = {"__name__": "__main__", "bash": bash}
    for raw in requests:
        request = json.loads(raw)
        with tempfile.NamedTemporaryFile(prefix="hero-python-") as capture:
            os.dup2(capture.fileno(), 1)
            os.dup2(capture.fileno(), 2)
            try:
                run_cell(request["code"], namespace)
            except KeyboardInterrupt:
                print("KeyboardInterrupt", file=sys.stderr)
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
            output = read_bounded(capture.name)
        protocol.write(json.dumps({"output": output}) + "\n")


if __name__ == "__main__":
    main()
