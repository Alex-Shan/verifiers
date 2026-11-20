"""The GEPA runner: split tasks, drive `gepa.api.optimize` against a `GEPAAdapter`, save
results. Unlike the async sibling runners, `optimize()` is synchronous and blocking, so it runs
on the main thread and the runner manages one event loop by hand: `env.serving()` is entered on
it once, each rollout batch runs via `loop.run_until_complete` (see `GEPAAdapter.evaluate`),
and everything is torn down in `finally`. Mirrors how v0 vf-gepa bridged sync GEPA to async
rollouts — a Ctrl-C raises on the main thread inside `optimize()` and unwinds through teardown."""

import asyncio
import random
from contextlib import ExitStack

from gepa.api import optimize
from gepa.core.result import GEPAResult
from gepa.strategies.proposal_sampling import PxNSampling
from gepa.utils import MaxCandidateProposalsStopper, NoImprovementStopper

from verifiers.v1.cli.output import output_path, save_config
from verifiers.v1.clients import ModelContext
from verifiers.v1.env import Env
from verifiers.v1.episode import Episode
from verifiers.v1.gepa.adapter import GEPAAdapter
from verifiers.v1.gepa.batch_sampler import CursorEpochShuffledBatchSampler
from verifiers.v1.gepa.config import GEPAConfig
from verifiers.v1.gepa.dataset import load_train_val_tasks, resolve_gepa_seed_prompt
from verifiers.v1.gepa.reflection import build_reflection_lm
from verifiers.v1.gepa.wandb_metrics import track_wandb_metrics
from verifiers.v1.utils.logging import get_logger
from verifiers.v1.utils.trace_store import append_episode

logger = get_logger(__name__)


class _GEPALog:
    """Forwards GEPA's optimizer log lines to v1 logging. GEPA's `LoggerProtocol` needs only
    `.log(str)`, so its per-iteration progress rides the same loguru setup as the rest of the
    v1 CLIs."""

    def log(self, message: str) -> None:
        logger.info(message)


def _stopping_kwargs(config: GEPAConfig) -> dict:
    if config.max_total_rollouts is not None:
        return {"max_metric_calls": config.max_total_rollouts}
    if config.max_iterations is not None:
        return {"stop_callbacks": MaxCandidateProposalsStopper(config.max_iterations)}
    assert config.max_iterations_without_improvement is not None
    return {
        "stop_callbacks": NoImprovementStopper(
            config.max_iterations_without_improvement
        )
    }


def run_gepa(env: Env, config: GEPAConfig) -> GEPAResult:
    logger.info("GEPA config:\n%s", config.model_dump_json(indent=2))
    train_tasks, val_tasks = load_train_val_tasks(
        env.taskset,
        config.select,
        config.num_train,
        config.num_val,
        config.train_tasks,
        config.val_tasks,
    )
    logger.info(
        "Loaded GEPA datasets train=%d validation=%d", len(train_tasks), len(val_tasks)
    )
    selected_tasks = [*train_tasks, *val_tasks]
    # Seed from the tasks GEPA actually evaluates (train ∪ val), not the full pre-split pool —
    # a taskset with per-task system prompts could otherwise seed from a task in neither split.
    seed_prompt = resolve_gepa_seed_prompt(selected_tasks, config.initial_prompt)
    tasks_by_idx = {task.data.idx: task for task in selected_tasks}

    run_dir = output_path(config) if config.save_results else None
    reflective_dataset_path = None
    if run_dir is not None:
        save_config(
            config, run_dir, "gepa.json"
        )  # resolved config + a fresh traces.jsonl
        reflective_dataset_path = run_dir / "reflective_dataset.jsonl"
        reflective_dataset_path.write_bytes(b"")
        logger.info("results: %s", run_dir)

    # optimize() is synchronous and blocking, so it drives the run from this (main) thread. We
    # own one event loop: `env.serving()` (shared tool servers + interception pool, built
    # once) is entered on it, each rollout batch runs on it via `loop.run_until_complete`
    # (GEPAAdapter.evaluate), and it's all torn down in `finally`. Keeping optimize() on the
    # main thread means a Ctrl-C raises straight through it into this teardown.
    loop = asyncio.new_event_loop()
    semaphore = (
        asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None
    )
    # Stream every rollout's episode to traces.jsonl as it finalizes — the same persist hook
    # `env.run_slot` takes (each trace records its candidate prompt).
    write_lock = asyncio.Lock()

    async def on_complete(episode: Episode) -> None:
        if run_dir is not None:
            await append_episode(run_dir, episode, write_lock)

    try:
        # The endpoint stays config: the interception server builds the live client.
        ctx = ModelContext(
            client=config.client, model=config.model, sampling=config.sampling
        )
        reflection_lm = build_reflection_lm(config)
        serving = env.serving()
        loop.run_until_complete(serving.__aenter__())
        # Pair __aexit__ with a *successful* __aenter__: this inner block is reached only once
        # serving is up, so a startup failure propagates as-is instead of being masked by
        # tearing down a context that never entered.
        try:
            adapter = GEPAAdapter(
                env=env,
                ctx=ctx,
                tasks=tasks_by_idx,
                loop=loop,
                semaphore=semaphore,
                on_complete=on_complete,
                reflection_columns=config.reflection_columns,
                reflective_dataset_path=reflective_dataset_path,
            )
            optimize_kwargs: dict = {
                "seed_candidate": {"system_prompt": seed_prompt},
                "trainset": [task.data.idx for task in train_tasks],
                "valset": [task.data.idx for task in val_tasks],
                "adapter": adapter,
                "reflection_lm": reflection_lm,
                **_stopping_kwargs(config),
                "reflection_minibatch_size": config.reflection_minibatch_size,
                "sampling_strategy": PxNSampling(
                    p=config.proposal_parents, n=config.mutations_per_parent
                ),
                "run_dir": str(run_dir) if run_dir is not None else None,
                "seed": config.seed,
                "display_progress_bar": False,
                "skip_perfect_score": False,
                "logger": _GEPALog(),
            }
            if config.reflection_batch_sampler == "cursor_epoch_shuffled":
                optimize_kwargs.update(
                    batch_sampler=CursorEpochShuffledBatchSampler(
                        config.reflection_minibatch_size, rng=random.Random(config.seed)
                    ),
                    reflection_minibatch_size=None,
                )
            with ExitStack() as tracking:
                if config.use_wandb:
                    optimize_kwargs["callbacks"] = [
                        tracking.enter_context(
                            track_wandb_metrics(
                                config, run_dir, len(train_tasks), len(val_tasks)
                            )
                        )
                    ]
                result = optimize(**optimize_kwargs)
            if run_dir is not None:
                # Persist the winning prompt as a plain file so it can be handed straight to
                # eval/train via `--env.taskset.system-prompt` (see TasksetConfig).
                candidate = result.best_candidate
                best = (
                    candidate.get("system_prompt", "")
                    if isinstance(candidate, dict)
                    else str(candidate)
                )
                best_path = run_dir / "best_system_prompt.txt"
                best_path.write_text(best, encoding="utf-8")
                logger.info("best system prompt: %s", best_path)
            return result
        finally:
            loop.run_until_complete(serving.__aexit__(None, None, None))
    finally:
        loop.close()
