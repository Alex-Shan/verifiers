"""检查镜像中预装的 agent 运行时和 ACP 适配器。"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

NODE = Path("/var/tmp/vf-node")


def command(*args: str) -> str:
    """运行命令并返回输出；失败时保留完整的诊断信息。"""
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def check_python_packages() -> None:
    packages = {
        "openai": ("openai", True, None),
        "mcp": ("mcp", True, "2.0.0"),
        "httpx": ("httpx", True, None),
        "httpx2": ("httpx2", True, None),
        "tenacity": ("tenacity", True, None),
        "agent-client-protocol": ("acp", True, "0.12.1"),
        "certifi": ("certifi", True, None),
    }
    for distribution, (module_name, importable, expected_version) in packages.items():
        installed_version = version(distribution)
        if expected_version is not None:
            assert installed_version == expected_version, (
                f"{distribution} version is {installed_version!r}, "
                f"expected {expected_version!r}"
            )
        if importable:
            importlib.import_module(module_name)
        print(f"  ✓ Python package {distribution} ({installed_version})")


def check_node() -> None:
    node = NODE / "bin/node"
    npm = NODE / "bin/npm"
    assert node.is_file() and node.stat().st_mode & 0o111, f"missing executable: {node}"
    assert npm.is_file() and npm.stat().st_mode & 0o111, f"missing executable: {npm}"
    assert command(str(node), "--version") == "v22.19.0"
    print("  ✓ Node.js v22.19.0")


def check_npm_install(
    name: str,
    directory: Path,
    expected_packages: dict[str, str],
    binaries: tuple[str, ...],
    ready: Path | None = None,
) -> None:
    assert directory.is_dir(), f"missing {name} directory: {directory}"
    if ready is not None:
        assert ready.is_file(), f"missing {name} ready marker: {ready}"

    package_root = directory / "node_modules"
    for package, expected_version in expected_packages.items():
        package_json = package_root / package / "package.json"
        assert package_json.is_file(), f"missing npm package: {package_json}"
        actual_version = json.loads(package_json.read_text())["version"]
        assert actual_version == expected_version, (
            f"{package} version is {actual_version!r}, expected {expected_version!r}"
        )

    for binary in binaries:
        path = package_root / ".bin" / binary
        assert path.is_file() and path.stat().st_mode & 0o111, (
            f"missing executable npm bin: {path}"
        )
    print(f"  ✓ {name} ({', '.join(expected_packages.values())})")


def check_agents() -> None:
    check_node()
    assert command("uv", "--version").startswith("uv "), "uv is unavailable"
    print(f"  ✓ uv ({command('uv', '--version')})")

    check_python_packages()

    hermes_dir = Path("/var/tmp/vf-hermes-agent-v2026.9.14")
    assert (hermes_dir / ".ready").is_file(), "Hermes setup would download its source"
    assert (hermes_dir / ".venv/bin/python").is_file(), "Hermes virtualenv is missing"
    hermes_version = command(
        "uv",
        "run",
        "--project",
        str(hermes_dir),
        "--offline",
        "--no-sync",
        "python",
        "-c",
        "from importlib.metadata import version; import hermes_cli; print(version('hermes-agent'))",
    )
    assert hermes_version == "0.21.3", hermes_version
    print(f"  ✓ Hermes Agent v2026.9.14 (package {hermes_version})")

    check_npm_install(
        "Claude Code ACP",
        Path("/var/tmp/vf-claude-agent-acp-2.1.278-0.79.0/packages"),
        {
            "@anthropic-ai/claude-code": "2.1.278",
            "@agentclientprotocol/claude-agent-acp": "0.79.0",
        },
        ("claude", "claude-agent-acp"),
        Path("/var/tmp/vf-claude-agent-acp-2.1.278-0.79.0/.ready"),
    )
    check_npm_install(
        "Codex ACP",
        Path("/var/tmp/vf-codex-0.155.1-1.12.0/acp"),
        {
            "@openai/codex": "0.155.1",
            "@agentclientprotocol/codex-acp": "1.12.0",
        },
        ("codex", "codex-acp"),
        Path("/var/tmp/vf-codex-0.155.1-1.12.0/.ready"),
    )

    pi_dir = Path("/var/tmp/vf-pi/mcp")
    assert (pi_dir / ".versions").read_text() == "0.86.1:2.34.0:0.0.33"
    check_npm_install(
        "Pi ACP",
        pi_dir,
        {
            "@earendil-works/pi-coding-agent": "0.86.1",
            "pi-mcp-adapter": "2.34.0",
            "pi-acp": "0.0.33",
        },
        ("pi", "pi-acp"),
    )

    # Each PEP 723 script gets a new path in the runtime. Resolve both dependency
    # sets from a fresh path with network disabled, as setup() has to do.
    scripts = {
        "Bash/Null": (
            '# requires-python = ">=3.10"',
            '# dependencies = ["openai", "mcp==2.0.0", "httpx", "httpx2", "tenacity", "certifi"]',
            "import openai, mcp, httpx, httpx2, tenacity, certifi",
        ),
        "ACP": (
            '# requires-python = ">=3.10,<3.15"',
            '# dependencies = ["agent-client-protocol==0.12.1", "httpx"]',
            "import acp, httpx",
        ),
    }
    with tempfile.TemporaryDirectory() as temporary:
        for name, (python_spec, dependencies, imports) in scripts.items():
            script = Path(temporary) / f"{name.lower().replace('/', '-')}.py"
            script.write_text(
                f"# /// script\n{python_spec}\n{dependencies}\n# ///\n{imports}\n"
            )
            environment = {**os.environ, "UV_OFFLINE": "true"}
            subprocess.run(
                ["uv", "sync", "--script", str(script), "--no-config"],
                check=True,
                env=environment,
            )
            subprocess.run(
                ["uv", "run", "--no-project", "--offline", str(script)],
                check=True,
                env=environment,
            )
            print(f"  ✓ {name} PEP 723 dependencies resolve offline")


def main() -> None:
    check_agents()
    print("Agent 安装检查通过（bash/null、Hermes、Claude Code、Codex、Pi）")


if __name__ == "__main__":
    main()
