"""Compare reverse-text GEPA runs with GR-GEPA disabled and enabled.

Run with ``uv run python tests/v1/run_reverse_text_gepa.py``. Add ``--dry-run``
to validate both configurations without making model calls.
"""

import argparse
import os
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs/gepa/reverse_text.toml"
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    api_key_var = config["client"]["api_key_var"]
    if not args.dry_run and not os.environ.get(api_key_var):
        parser.error(f"set {api_key_var} before running the experiments")

    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{os.getpid()}"
    # for enabled in (False, True):
    enabled = True
    if enabled:
        mode = "on" if enabled else "off"
        run_name = f"reverse-text-gepa-{run_id}-{mode}"
        print(f"Running {mode}: {config['output_dir']}/{run_name}", flush=True)
        command = [
            "uv",
            "run",
            "--no-sync",
            "vf-gepa",
            "@",
            str(config_path),
            "--gr-gepa",
            str(enabled),
            "--group-size",
            "4",
            "--group-alpha",
            "0.8",
            "--group-success-threshold",
            "0.8",
            "--run.name",
            run_name,
        ]
        if args.dry_run:
            command.extend(["--dry-run", "True"])
        subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
