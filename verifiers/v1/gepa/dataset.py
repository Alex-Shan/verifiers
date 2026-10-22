"""Train/val split and upfront validation for a GEPA run.

v1 tasksets have no generic train/val split concept (`TasksetConfig` has no `split` field;
individual tasksets define ad hoc ones inconsistently), so GEPA carves one out of the tasks
`select` yields (reproducible across runs like every other entrypoint): two
disjoint slices.
"""

from pathlib import Path

from verifiers.v1.configs.select import SelectConfig
from verifiers.v1.task import Task
from verifiers.v1.taskset import Taskset


def load_tasks_from_path(
    taskset: Taskset, path: Path, select: SelectConfig
) -> list[Task]:
    """Load one explicit dataset through the configured taskset's normal parser.

    Separate train/validation files are supported by tasksets whose config exposes a
    `data_path` field. Reconstructing the same taskset with that path preserves its
    task conversion, task config, and reward implementation.
    """
    if "data_path" not in type(taskset.config).model_fields:
        raise ValueError(
            "train_tasks/val_tasks require a taskset config with a `data_path` field"
        )
    if not path.is_file():
        raise FileNotFoundError(f"GEPA dataset file not found: {path.resolve()}")

    config = taskset.config.model_copy(update={"data_path": str(path)})
    tasks = list(type(taskset)(config).select(select))
    if not tasks:
        raise ValueError(f"GEPA dataset selection produced no tasks: {path}")
    return tasks


def split_tasks(
    tasks: list[Task], num_train: int, num_val: int
) -> tuple[list[Task], list[Task]]:
    """The selected tasks, split into disjoint `(train, val)` slices (`train` feeds reflection
    minibatches; `val` scores each candidate for the pareto frontier)."""
    if num_train + num_val > len(tasks):
        raise ValueError(
            f"requested {num_train} train + {num_val} val tasks, but the taskset only "
            f"loaded {len(tasks)}"
        )
    return tasks[:num_train], tasks[num_train : num_train + num_val]


def resolve_gepa_seed_prompt(tasks: list[Task], initial_prompt: str | None) -> str:
    """The system prompt GEPA starts optimizing from: `initial_prompt` if given, else the first
    task that sets `system_prompt`. Some tasksets (e.g. `gsm8k`) bake instructions into
    `prompt` rather than `system_prompt` and can't be optimized this way — pass `--initial-prompt`
    to seed one explicitly.

    How the resolved prompt reaches the model at rollout time — a real system message vs. folded
    into the user prompt — is the harness's call: `Harness.resolve_prompt` owns that policy and
    warns when it folds, so it isn't re-policed here."""
    if initial_prompt is not None:
        return initial_prompt
    for task in tasks:
        if task.data.system_prompt is not None:
            return task.data.system_prompt
    raise ValueError(
        "no task in this taskset sets Task.system_prompt — some tasksets bake instructions "
        "directly into `prompt` instead (e.g. gsm8k) and can't be optimized this way. Pass "
        "--initial-prompt to seed one explicitly, or pick a taskset whose load() sets "
        "system_prompt on its task data (e.g. reverse-text, textarena)."
    )
