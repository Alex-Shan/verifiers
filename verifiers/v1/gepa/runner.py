"""The GEPA runner: split tasks, drive `gepa.api.optimize` against a `GEPAAdapter`, save
results. Unlike the async sibling runners, `optimize()` is synchronous and blocking, so it runs
on the main thread and the runner manages one event loop by hand: `env.serving()` is entered on
it once, each rollout batch runs via `loop.run_until_complete` (see `GEPAAdapter.evaluate`),
and everything is torn down in `finally`. Mirrors how v0 vf-gepa bridged sync GEPA to async
rollouts — a Ctrl-C raises on the main thread inside `optimize()` and unwinds through teardown."""

import asyncio
import contextlib
import fcntl
import hashlib
import json
import random
from uuid import uuid4

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
from verifiers.v1.gepa.dataset import (
    load_tasks_from_path,
    resolve_gepa_seed_prompt,
    split_tasks,
)
from verifiers.v1.gepa.reflection import (
    EFFICIENCY_REFLECTION_TEMPLATE,
    TOKEN_REFLECTION_TEMPLATE,
    build_reflection_lm,
)
from verifiers.v1.utils.logging import get_logger
from verifiers.v1.utils.trace_store import append_episode

logger = get_logger(__name__)


class _GEPALog:
    """Forwards GEPA's optimizer log lines to v1 logging. GEPA's `LoggerProtocol` needs only
    `.log(str)`, so its per-iteration progress rides the same loguru setup as the rest of the
    v1 CLIs."""

    def log(self, message: str) -> None:
        logger.info(message)


@contextlib.contextmanager
def _calibration_lock(path):
    """Serialize shared scale initialization across concurrent experiment arms."""
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _stopping_kwargs(config: GEPAConfig) -> dict:
    if config.max_total_rollouts is not None:
        return {"max_metric_calls": config.max_total_rollouts}
    if config.max_iterations is not None:
        # This stopper checks GEPA's state.i, which counts completed loop
        # iterations independently of training rollout group size.
        return {"stop_callbacks": MaxCandidateProposalsStopper(config.max_iterations)}
    assert config.max_iterations_without_improvement is not None
    return {
        "stop_callbacks": NoImprovementStopper(
            config.max_iterations_without_improvement
        )
    }


def run_gepa(env: Env, config: GEPAConfig) -> GEPAResult:
    logger.info("gepa config:\n%s", config.model_dump_json(indent=2))
    if config.train_tasks is None:
        taskset = env.taskset.select(config.select)
        all_tasks = list(taskset.take(config.num_train + config.num_val))
        train_tasks, val_tasks = split_tasks(
            all_tasks, config.num_train, config.num_val
        )
    else:
        # Explicit files define the two complete datasets. Load both through the
        # same taskset implementation so their rows receive the normal task parsing
        # and scoring configuration.
        assert config.val_tasks is not None
        train_tasks = load_tasks_from_path(
            env.taskset, config.train_tasks, config.select
        )
        val_tasks = load_tasks_from_path(env.taskset, config.val_tasks, config.select)

        # Each independently loaded taskset numbers its rows from zero. GEPA keys
        # tasks by TaskData.idx, so move validation indices past the train indices.
        next_idx = (
            max(
                (task.data.idx for task in train_tasks if task.data.idx is not None),
                default=-1,
            )
            + 1
        )
        val_tasks = [
            task.with_data(idx=next_idx + i) for i, task in enumerate(val_tasks)
        ]

    logger.info(
        "GEPA dataset sizes: %d train, %d validation",
        len(train_tasks),
        len(val_tasks),
    )
    selected_tasks = [*train_tasks, *val_tasks]
    # Seed from the tasks GEPA actually evaluates (train ∪ val), not the full pre-split pool —
    # a taskset with per-task system prompts could otherwise seed from a task in neither split.
    seed_prompt = resolve_gepa_seed_prompt(selected_tasks, config.initial_prompt)
    tasks_by_idx = {task.data.idx: task for task in selected_tasks}

    run_dir = output_path(config) if config.save_results else None
    reflective_dataset_path = (
        run_dir / "reflective_dataset.jsonl" if run_dir is not None else None
    )

    wandb_dir = None
    if config.use_wandb and run_dir is not None:
        wandb_dir = run_dir / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)

    if run_dir is not None:
        import json as recovery_json

        identity = config.model_dump(mode="json")
        identity.pop("resume", None)
        manifest = run_dir / "recovery_identity.json"
        if config.resume:
            if (
                not manifest.exists()
                or recovery_json.loads(manifest.read_text()) != identity
            ):
                raise ValueError("Recovery identity missing or mismatched")
        elif manifest.exists() or (run_dir / "gepa_state.bin").exists():
            raise ValueError("Run already exists; use resume or a new run name")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest.write_text(recovery_json.dumps(identity, indent=2))
    if run_dir is not None and not config.resume:
        save_config(
            config, run_dir, "gepa.json"
        )  # resolved config + a fresh traces.jsonl
        reflective_dataset_path.write_text("", encoding="utf-8")
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
                train_task_ids={task.data.idx for task in train_tasks},
                semaphore=semaphore,
                on_complete=on_complete,
                reflection_columns=config.reflection_columns,
                reflective_dataset_path=reflective_dataset_path,
                efficiency_protocol=config.efficiency_protocol,
                efficiency_feedback=config.efficiency_feedback,
                efficiency_objective=config.efficiency_objective,
                evaluation_metrics_path=run_dir / "evaluation_metrics.jsonl"
                if run_dir
                else None,
                validation_rollouts=config.validation_rollouts,
                gr_gepa=config.gr_gepa,
                group_size=config.group_size,
                group_alpha=config.group_alpha,
                group_time_weight=config.group_time_weight,
                group_success_threshold=config.group_success_threshold,
            )
            adapter.recovery_dir = run_dir / "recovery" if run_dir else None
            from verifiers.v1.harnesses.hero_python.harness import (
                HeroPythonHarnessConfig,
            )

            hero_python = (
                isinstance(
                    getattr(config.env, "agent", None).harness, HeroPythonHarnessConfig
                )
                if getattr(config.env, "agent", None)
                else False
            )
            if config.efficiency_objective or adapter.cost_feedback or hero_python:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    config.model, local_files_only=True
                )
                adapter.count_prompt_tokens = lambda text: len(
                    tokenizer.encode(text, add_special_tokens=False)
                )
            if hero_python:
                from verifiers.v1.harnesses.hero_python.prompt import (
                    HERO_PYTHON_SYSTEM_PROMPT,
                )

                adapter.system_prompt_prefix = HERO_PYTHON_SYSTEM_PROMPT
            if config.efficiency_objective or adapter.cost_feedback:
                calibration_path = config.efficiency_calibration_path
                identity = {
                    "model": config.model,
                    "seed": config.seed,
                    "prompt": seed_prompt,
                    "sampling": config.sampling.model_dump(mode="json"),
                    "env": config.env.model_dump(mode="json"),
                    "validation": hashlib.sha256(
                        repr([(t.data.idx, t.data.prompt) for t in val_tasks]).encode()
                    ).hexdigest(),
                }
                if config.efficiency_protocol != "legacy":
                    identity["efficiency_protocol"] = config.efficiency_protocol
                identity["validation_rollouts"] = config.validation_rollouts
                identity["validation_protocol"] = "mean-pass1-v2"
                identity["effective_env"] = config.env.model_dump(mode="json")
                with _calibration_lock(calibration_path):
                    if calibration_path is not None and calibration_path.exists():
                        calibration = json.loads(calibration_path.read_text())
                        if calibration["identity"] != identity:
                            raise ValueError(
                                "Shared calibration does not match this experiment setup"
                            )
                        adapter.efficiency_scales = calibration["scales"]
                        expected_scales = {"prompt", "tokens"}
                        if config.efficiency_protocol == "legacy":
                            expected_scales.add("agent_seconds")
                        if set(adapter.efficiency_scales) != expected_scales or any(
                            not isinstance(v, (int, float)) or not 0 < v < float("inf")
                            for v in adapter.efficiency_scales.values()
                        ):
                            raise ValueError(
                                "Shared calibration scales must be finite and positive"
                            )
                    else:
                        if (
                            config.max_total_rollouts is not None
                            and config.max_total_rollouts
                            < len(val_tasks) * config.validation_rollouts
                        ):
                            raise ValueError(
                                "Rollout budget must cover seed validation calibration"
                            )
                        adapter.calibrate(
                            [task.data.idx for task in val_tasks], seed_prompt
                        )
                        if calibration_path is not None:
                            calibration_path.parent.mkdir(parents=True, exist_ok=True)
                            with calibration_path.open("x", encoding="utf-8") as file:
                                json.dump(
                                    {
                                        "identity": identity,
                                        "scales": adapter.efficiency_scales,
                                    },
                                    file,
                                    indent=2,
                                )
                if run_dir is not None:
                    (run_dir / "efficiency_scales.json").write_text(
                        json.dumps(adapter.efficiency_scales, indent=2),
                        encoding="utf-8",
                    )
            sampler = CursorEpochShuffledBatchSampler(
                config.reflection_minibatch_size, rng=random.Random(config.seed)
            )
            adapter.sampler = sampler
            adapter.teacher = reflection_lm
            adapter.recovery_dir = run_dir / "recovery" if run_dir else None
            from gepa.strategies.candidate_selector import ParetoCandidateSelector

            optimize_kwargs: dict = {
                "candidate_selection_strategy": ParetoCandidateSelector(
                    rng=sampler.rng
                ),
                "seed_candidate": {"system_prompt": seed_prompt},
                "trainset": [task.data.idx for task in train_tasks],
                "valset": [task.data.idx for task in val_tasks],
                "adapter": adapter,
                "reflection_lm": reflection_lm,
                # GEPA requires this to be None when passing a custom sampler.
                "reflection_minibatch_size": None,
                "batch_sampler": sampler,
                "sampling_strategy": PxNSampling(p=2, n=2),
                "run_dir": str(run_dir) if run_dir is not None else None,
                "seed": config.seed,
                "display_progress_bar": False,
                "skip_perfect_score": False
                if config.efficiency_protocol == "token_v1"
                else not config.efficiency_objective,
                "perfect_score": 1.0,
                "logger": _GEPALog(),
                "use_wandb": config.use_wandb,
                "wandb_init_kwargs": {
                    "id": uuid4().hex,
                    "resume": "never",
                    "project": config.wandb_project,
                    "name": config.run.name,
                    "dir": str(wandb_dir) if wandb_dir is not None else None,
                    "config": {
                        "model": config.model,
                        "num_train": len(train_tasks),
                        "num_val": len(val_tasks),
                        "proposal_parents": 2,
                        "mutations_per_parent": 2,
                        "batch_sampler": "cursor_epoch_shuffled",
                        "reflection_minibatch_size": config.reflection_minibatch_size,
                        "max_total_rollouts": config.max_total_rollouts,
                        "max_iterations": config.max_iterations,
                        "max_iterations_without_improvement": config.max_iterations_without_improvement,
                        "seed": config.seed,
                        "efficiency_protocol": config.efficiency_protocol,
                        "efficiency_objective": config.efficiency_objective,
                        "efficiency_feedback": adapter.cost_feedback,
                    },
                },
            }
            if adapter.cost_feedback:
                optimize_kwargs["reflection_prompt_template"] = (
                    TOKEN_REFLECTION_TEMPLATE
                    if config.efficiency_protocol == "token_v1"
                    else EFFICIENCY_REFLECTION_TEMPLATE
                )
            optimize_kwargs.update(_stopping_kwargs(config))
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
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(serving.__aexit__(None, None, None))
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
