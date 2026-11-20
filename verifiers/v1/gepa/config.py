"""The `GEPAConfig`: the single config object the `gepa` CLI parses.

GEPA optimizes one taskset's `Task.system_prompt` by alternating rollouts (`evaluate`) with a
teacher LM reflecting on the reflective dataset (`make_reflective_dataset`) — see
`verifiers.v1.gepa.adapter.GEPAAdapter`. It owns an `env` field (the
environment: its taskset, seats, limits) and adds the optimization loop's own knobs (model,
reflection model, train/val split, budget). GEPA always runs in-process, since its adapter protocol is itself synchronous (see
`GEPAAdapter`)."""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import EvalClientConfig
from verifiers.v1.configs.cli.env import narrowed_env_annotation, resolve_env_field
from verifiers.v1.configs.cli.run import RunConfig, default_run_name
from verifiers.v1.configs.env import EnvConfig
from verifiers.v1.configs.sampling import SamplingConfig
from verifiers.v1.configs.select import SelectCLIConfig
from verifiers.v1.envs.single_agent import SingleAgentEnvConfig


class GEPAConfig(BaseConfig):
    """The GEPA run plus its environment. `model` runs the rollouts under optimization;
    `reflection_model` (defaults to `model`) proposes new system prompts from the reflective
    dataset."""

    env: SerializeAsAny[EnvConfig] = SingleAgentEnvConfig()
    """The environment under optimization — the same `[env]` block as an eval's
    (`--env.taskset.*`, seats, limits), narrowed to the selected env's config class."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_env(cls, data):
        return resolve_env_field(data, narrowed_env_annotation(cls))

    run: RunConfig = Field(default_factory=RunConfig)
    """Run identity: `run.name` is the display name and `run.dir` names the directory
    under `output_dir`; both auto-generate like an eval's when unset."""
    model: str = Field(
        "deepseek/deepseek-v4-flash", validation_alias=AliasChoices("model", "m")
    )
    """Model id for rollouts under optimization."""
    client: EvalClientConfig = EvalClientConfig()
    sampling: SamplingConfig = SamplingConfig()
    reflection_model: str | None = None
    """Teacher model that proposes new system prompts. None = reuse `model`."""
    reflection_client: EvalClientConfig | None = None
    """Endpoint for `reflection_model`. None = reuse `client`."""

    select: SelectCLIConfig = SelectCLIConfig()
    """Which of the taskset's tasks GEPA splits into train/val, under `--select.*`
    (`-n` sets `select.limit`, `-s` sets `select.shuffle`)."""
    train_tasks: Path | None = None
    """A training JSONL file or glob. Set together with val_tasks."""
    val_tasks: Path | None = None
    """A validation JSONL file or glob. Set together with train_tasks."""
    num_train: int = Field(100, ge=1)
    """Tasks reserved for reflection minibatches (GEPA never scores the full trainset at once)."""
    num_val: int = Field(50, ge=1)
    """Tasks held out to score each candidate system prompt for the pareto frontier."""
    seed: int = 0
    """Seed for GEPA's optimizer (candidate selection / minibatch sampling). The train/val
    split comes from `select` (`-s` shuffles it under `select.seed`), so this doesn't
    change it."""

    max_total_rollouts: int | None = Field(None, ge=1)
    """Rollout threshold checked between iterations. Defaults to 500 if no stop is set."""
    max_iterations: int | None = Field(
        None, ge=1, validation_alias=AliasChoices("max_iterations", "max_iteration")
    )
    """Maximum number of GEPA iterations after the seed validation."""
    max_iterations_without_improvement: int | None = Field(None, ge=1)
    """Stop after this many iterations without a better validation score."""
    reflection_minibatch_size: int = Field(3, ge=1)
    """Train tasks sampled per proposal."""
    reflection_batch_sampler: Literal["epoch_shuffled", "cursor_epoch_shuffled"] = (
        "epoch_shuffled"
    )
    proposal_parents: int = Field(1, ge=1)
    mutations_per_parent: int = Field(1, ge=1)
    reflection_columns: list[str] = Field(default_factory=list)
    """Extra per-trace fields (from `trace.info`, else `task`) to surface to the teacher LM."""
    initial_prompt: str | None = None
    """Seed system prompt. None = the first loaded task's `Task.system_prompt`, if any task
    sets one (see `resolve_gepa_seed_prompt`)."""

    max_concurrent: int | None = Field(
        128, validation_alias=AliasChoices("max_concurrent", "c")
    )
    """Max rollouts in flight at once, across the whole run."""
    output_dir: Path = Field(
        Path("outputs"), validation_alias=AliasChoices("output_dir", "o")
    )
    """Directory that groups related runs. The run (`configs/gepa.json` + the streamed
    `traces.jsonl`, alongside GEPA's own `candidates.json` / `run_log.json`) writes to
    `output_dir / run.dir`."""
    save_results: bool = True
    clean: bool = Field(False, exclude=True)
    """Delete the selected run directory before starting a fresh run."""
    use_wandb: bool = False
    wandb_project: str = "GEPA"
    verbose: bool = Field(False, validation_alias=AliasChoices("verbose", "v"))
    dry_run: bool = Field(False, exclude=True)
    """Resolve + validate the config and dump it, then exit. Excluded from the
    saved config so re-running `@ configs/gepa.json` runs for real."""

    @model_validator(mode="after")
    def validate_run(self):
        if (self.train_tasks is None) != (self.val_tasks is None):
            raise ValueError("train_tasks and val_tasks must be set together")
        if (
            self.train_tasks is not None
            and "data_path" not in type(self.env.taskset).model_fields
        ):
            raise ValueError("train_tasks/val_tasks require a taskset with data_path")

        stop_values = (
            self.max_total_rollouts,
            self.max_iterations,
            self.max_iterations_without_improvement,
        )
        stop_count = sum(value is not None for value in stop_values)
        if stop_count > 1:
            raise ValueError(
                "Set only one of max_total_rollouts, max_iterations, "
                "or max_iterations_without_improvement"
            )
        if stop_count == 0:
            self.max_total_rollouts = 500

        if self.run.name is None:
            self.run.name = default_run_name(self.env, self.model)
        if self.run.dir is None:
            self.run.dir = self.run.name
        return self
