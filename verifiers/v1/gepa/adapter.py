"""The GEPA <-> v1 bridge: run a candidate system prompt over a batch of tasks and score it.

GEPA's adapter protocol (`evaluate`, `make_reflective_dataset`) is synchronous and
`gepa.api.optimize` blocks, but v1 rollouts are async. The runner manages one event loop by
hand: it enters `env.serving()` on that loop and runs the blocking `optimize()` on the main
thread, so each synchronous `evaluate()` drives its batch of rollouts with
`loop.run_until_complete` — the one sync↔async hop. (Mirrors how v0 vf-gepa bridged; no worker
thread, so a Ctrl-C unwinds straight through `optimize()` into the runner's teardown.)
"""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from gepa.core.adapter import EvaluationBatch
from pydantic_core import to_jsonable_python

from verifiers.v1.clients import ModelContext
from verifiers.v1.env import Env
from verifiers.v1.episode import Episode
from verifiers.v1.task import Task

Candidate = dict[str, str]
Trajectory = Episode | list[Episode]
_EPSILON = 1e-8


@dataclass
class GEPAAdapter:
    """Bridges GEPA's optimization loop with a native v1 `Env`. `tasks` covers only the
    trainset + valset tasks GEPA was given (not the whole taskset), keyed by `task.data.idx` —
    GEPA's `batch` is a list of those idxs, and injecting the candidate rebuilds each `Task`
    around a `data` row carrying the new `system_prompt`. `loop` is the runner's persistent event
    loop (which holds `env.serving()` open); `evaluate` drives its rollouts on it with
    `run_until_complete`."""

    env: Env
    ctx: ModelContext
    tasks: dict[int, Task]
    loop: asyncio.AbstractEventLoop
    train_task_ids: set[int] = field(default_factory=set)
    semaphore: asyncio.Semaphore | None = None
    on_complete: Callable[[Episode], Awaitable[None]] | None = None
    """Called with each rollout's episode as it finalizes — the runner's persist hook that
    streams episodes to `traces.jsonl`."""
    reflection_columns: list[str] = field(default_factory=list)
    propose_new_texts: Callable[..., Candidate] | None = None
    """Part of GEPA's adapter protocol — its proposer reads this attribute on every reflection
    step. None = use GEPA's default reflection-LM proposer (the AttributeError from leaving it
    undeclared silently disables all mutation proposals)."""
    reflective_dataset_path: Path | None = None
    system_prompt_prefix: str = ""
    efficiency_objective: bool = False
    count_prompt_tokens: Callable[[str], int] | None = None
    efficiency_scales: dict[str, float] = field(default_factory=dict)
    calibration_prompt: str | None = None
    calibration_cache: dict[int, Episode | list[Episode]] = field(default_factory=dict)
    evaluation_metrics_path: Path | None = None
    validation_rollouts: int = 1
    gr_gepa: bool = False
    group_size: int = 4
    group_alpha: float = 0.5
    group_time_weight: float = 0.5
    group_success_threshold: float = 0.9

    def evaluate(
        self,
        batch: list[int],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, Trajectory]:
        """Run `candidate`'s system prompt on the tasks named by `batch` (`Task.idx` values)
        and score them — one output and one score per batch entry, so the batch stays
        aligned even when an output contains several rollouts. Called synchronously by GEPA on
        the main thread; each batch's rollouts run on the runner's persistent loop via
        `run_until_complete`."""
        system_prompt = candidate.get("system_prompt", "")
        counts = [
            (self.group_size if self.gr_gepa else 1)
            if idx in self.train_task_ids
            else self.validation_rollouts
            for idx in batch
        ]
        if system_prompt == self.calibration_prompt and all(
            idx in self.calibration_cache and n == self.validation_rollouts
            for idx, n in zip(batch, counts)
        ):
            cached = [self.calibration_cache.pop(idx) for idx in batch]
            episodes = [
                e
                for item in cached
                for e in (item if isinstance(item, list) else [item])
            ]
        else:
            episodes = self.loop.run_until_complete(
                self._run_batch(batch, system_prompt, counts)
            )
        # Infrastructure failures must behave identically in every scoring mode.
        for episode in episodes:
            _episode_score(episode)
        if self.evaluation_metrics_path is not None:
            prompt_id = hashlib.sha256(system_prompt.encode()).hexdigest()
            with self.evaluation_metrics_path.open("a", encoding="utf-8") as file:
                sample_counts = {}
                for episode in episodes:
                    task_id = episode.task.data.idx
                    sample_counts[task_id] = sample_counts.get(task_id, 0) + 1
                    usage = getattr(episode, "usage", None)
                    row = {
                        "prompt_id": prompt_id,
                        "system_prompt": system_prompt,
                        "task_idx": episode.task.data.idx,
                        "sample_idx": sample_counts[task_id],
                        "split": "train"
                        if episode.task.data.idx in self.train_task_ids
                        else "validation",
                        "reward": _episode_score(episode),
                        "num_total_tokens": episode.num_total_tokens,
                        "provider_total_tokens": getattr(usage, "total_tokens", None),
                        "timing": _episode_timing(episode),
                    }
                    if self.count_prompt_tokens is not None:
                        row["prompt_tokens"] = self.count_prompt_tokens(system_prompt)
                        # Other harnesses may insert their own unmeasured system prompts.
                        row["full_system_prompt_tokens"] = (
                            self.count_prompt_tokens(
                                "\n\n".join(
                                    p
                                    for p in (self.system_prompt_prefix, system_prompt)
                                    if p
                                )
                            )
                            if self.system_prompt_prefix
                            else None
                        )
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs: list[Trajectory] = []
        scores: list[float] = []
        offset = 0
        for task_idx, n in zip(batch, counts):
            group = episodes[offset : offset + n]
            offset += n
            if task_idx not in self.train_task_ids and n > 1:
                accuracy = sum(_episode_score(e) for e in group) / n
                tokens = sum(e.num_total_tokens for e in group) / n
                seconds = (
                    sum(_episode_timing(e)["agent"]["duration"] for e in group) / n
                )
                value = accuracy
                if self.efficiency_objective:
                    scales = self.efficiency_scales
                    value *= (
                        0.8
                        + 0.05
                        / (
                            1
                            + self.count_prompt_tokens(system_prompt) / scales["prompt"]
                        )
                        + 0.075 / (1 + tokens / scales["tokens"])
                        + 0.075 / (1 + seconds / scales["agent_seconds"])
                    )
                outputs.append(group)
                scores.append(value)
                if self.evaluation_metrics_path is not None:
                    record = {
                        "prompt_id": hashlib.sha256(system_prompt.encode()).hexdigest(),
                        "task_idx": task_idx,
                        "n": n,
                        "c": sum(_episode_score(e) for e in group),
                        "pass_at_1": accuracy,
                        "pass_at_4": (float(accuracy > 0) if n == 4 else None),
                        "all_wrong": accuracy == 0,
                        "mixed": 0 < accuracy < 1,
                        "all_correct": accuracy == 1,
                        "tokens_mean": tokens,
                        "agent_seconds_mean": seconds,
                        "score": value,
                        "prompt_tokens": self.count_prompt_tokens(system_prompt)
                        if self.count_prompt_tokens
                        else None,
                    }
                    with self.evaluation_metrics_path.with_name(
                        "validation_metrics.jsonl"
                    ).open("a") as file:
                        file.write(json.dumps(record) + "\n")
            elif self.efficiency_objective:
                outputs.append(group if n > 1 else group[0])
                scores.append(
                    sum(self.efficiency_score(e, system_prompt) for e in group) / n
                )
            elif n > 1:
                outputs.append(group)
                scores.append(
                    sum(
                        _group_scores(
                            group,
                            self.group_alpha,
                            self.group_success_threshold,
                            self.group_time_weight,
                        )
                    )
                    / n
                )
            else:
                outputs.append(group[0])
                scores.append(_episode_score(group[0]))
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=outputs if capture_traces else None,
            num_metric_calls=len(episodes),
        )

    def calibrate(self, batch: list[int], system_prompt: str) -> None:
        """Freeze seed-validation scales and reuse these episodes in GEPA's initial eval."""
        if not self.efficiency_objective or self.count_prompt_tokens is None:
            raise ValueError("Efficiency calibration requires a prompt tokenizer")
        episodes = self.loop.run_until_complete(
            self._run_batch(
                batch, system_prompt, [self.validation_rollouts] * len(batch)
            )
        )
        for episode in episodes:
            _episode_score(episode)
        groups = [
            episodes[i : i + self.validation_rollouts]
            for i in range(0, len(episodes), self.validation_rollouts)
        ]
        lengths = [sum(e.num_total_tokens for e in g) / len(g) for g in groups]
        lengths = [v for v in lengths if v > 0]
        times = [
            sum(_episode_timing(e)["agent"]["duration"] for e in g) / len(g)
            for g in groups
        ]
        times = [t for t in times if t > 0]
        if not lengths or not times:
            raise ValueError(
                "Calibration requires positive token and agent-time measurements"
            )
        self.efficiency_scales = {
            "prompt": max(1, self.count_prompt_tokens(system_prompt)),
            "tokens": median(lengths),
            "agent_seconds": median(times),
        }
        self.calibration_prompt = system_prompt
        self.calibration_cache = dict(
            zip(
                batch,
                [g[0] if self.validation_rollouts == 1 else g for g in groups],
                strict=True,
            )
        )

    def efficiency_score(self, episode: Episode, system_prompt: str) -> float:
        if self.count_prompt_tokens is None or not self.efficiency_scales:
            raise ValueError("Efficiency scoring requires frozen calibration scales")
        reward = _episode_score(episode)
        if reward not in (0.0, 1.0):
            raise ValueError("Efficiency objective requires binary task rewards")
        prompt = self.count_prompt_tokens(system_prompt)
        tokens = episode.num_total_tokens
        seconds = _episode_timing(episode)["agent"]["duration"]
        return reward * (
            0.8
            + 0.05 / (1 + prompt / self.efficiency_scales["prompt"])
            + 0.075 / (1 + tokens / self.efficiency_scales["tokens"])
            + 0.075 / (1 + seconds / self.efficiency_scales["agent_seconds"])
        )

    async def _run_batch(
        self, batch: list[int], system_prompt: str, counts: list[int]
    ) -> list[Episode]:
        # Inject the candidate as a copy of each base task with its system_prompt overridden
        # (`with_system_prompt` copies rather than reconstructs, so subclass state survives and
        # the shared base task in `self.tasks` is left untouched for the next candidate).
        tasks = [self.tasks[idx].with_system_prompt(system_prompt) for idx in batch]
        slots = [
            slot
            for task, n in zip(tasks, counts, strict=True)
            for slot in self.env.slots(task, n=n)
        ]
        results = await asyncio.gather(
            *(
                self.env.run_slot(slot, self.ctx, self.semaphore, self.on_complete)
                for slot in slots
            )
        )
        return list(results)

    def make_reflective_dataset(
        self,
        candidate: Candidate,  # Required by GEPA's adapter protocol.
        eval_batch: EvaluationBatch[Trajectory, Trajectory],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build one reflective record per task group, or per trace in standard GEPA."""
        trajectories = eval_batch.trajectories or []
        records: list[dict[str, Any]] = []
        for trajectory in trajectories:
            if isinstance(trajectory, list) or self.efficiency_objective:
                trajectory = (
                    trajectory if isinstance(trajectory, list) else [trajectory]
                )
                scores = (
                    [
                        self.efficiency_score(e, candidate.get("system_prompt", ""))
                        for e in trajectory
                    ]
                    if self.efficiency_objective
                    else _group_scores(
                        trajectory,
                        self.group_alpha,
                        self.group_success_threshold,
                        self.group_time_weight,
                    )
                )
                rollouts = []
                for episode, score in zip(trajectory, scores, strict=True):
                    rollouts.append(
                        {
                            "reward": _episode_score(episode),
                            "num_total_tokens": episode.num_total_tokens,
                            "timing": _episode_timing(episode),
                            "score": score,
                            "traces": self._episode_records(episode),
                            "error": str(episode.last_error)
                            if episode.last_error
                            else None,
                        }
                    )
                records.append(
                    {
                        "comparison_summary": {
                            "successes": sum(r["reward"] == 1 for r in rollouts),
                            "failures": sum(r["reward"] == 0 for r in rollouts),
                            "outcomes_comparable": any(
                                r["reward"] == 1 for r in rollouts
                            )
                            and any(r["reward"] == 0 for r in rollouts),
                            "successful_costs": [
                                {
                                    "sample_idx": i + 1,
                                    "tokens": r["num_total_tokens"],
                                    "seconds": r["timing"]["agent"]["duration"],
                                }
                                for i, r in enumerate(rollouts)
                                if r["reward"] == 1
                            ],
                            "success_cost_comparison_available": sum(
                                r["reward"] == 1 for r in rollouts
                            )
                            > 1,
                        },
                        "comparison_guidance": (
                            EFFICIENCY_GUIDANCE
                            if self.efficiency_objective
                            else "Compare repeated rollouts of the same task. Diagnose recurring errors; "
                            "score combines success and within-group relative tokens and agent time. "
                            "Propose a generalizable prompt revision."
                        ),
                        "efficiency_scales": self.efficiency_scales,
                        "parent_prompt_tokens": self.count_prompt_tokens(
                            candidate.get("system_prompt", "")
                        )
                        if self.count_prompt_tokens
                        else None,
                        "parent_prompt": candidate.get("system_prompt", ""),
                        "query": to_jsonable_python(trajectory[0].task.data.prompt),
                        "rollouts": rollouts,
                    }
                )
            else:
                records.extend(self._episode_records(trajectory))
        if self.reflective_dataset_path is not None:
            with self.reflective_dataset_path.open("a", encoding="utf-8") as file:
                for record in records:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {comp: records for comp in components_to_update}

    def _episode_records(self, episode: Episode) -> list[dict[str, Any]]:
        records = []
        for trace in episode.traces:
            record: dict[str, Any] = {
                "query": to_jsonable_python(trace.task.data.prompt),
                "completion": [
                    to_jsonable_python(node.message) for node in trace.nodes
                ],
                "reward": trace.reward,
                "agent": trace.agent.name,
                "timing": _trace_timing(trace),
            }
            if trace.has_error:
                record["error"] = str(trace.last_error)
            if trace.stop_condition:
                record["stop_condition"] = trace.stop_condition
            for column in self.reflection_columns:
                if column in trace.info:
                    record[column] = to_jsonable_python(trace.info[column])
                elif hasattr(trace.task.data, column):
                    record[column] = to_jsonable_python(
                        getattr(trace.task.data, column)
                    )
            records.append(record)
        return records


def _episode_score(episode: Episode) -> float:
    """A candidate's score on one episode: the mean reward of the episode's scored
    traces. Seats that recorded no rewards (a reward-less judge) don't dilute the
    signal; a successful episode with no scored traces scores 0."""
    if any(trace.is_dropped for trace in episode.traces):
        raise ValueError("GEPA cannot score an episode containing dropped traces")
    if not episode.ok:
        raise ValueError(f"GEPA cannot score failed episode {episode.id!r}")
    scored = [trace.reward for trace in episode.traces if trace.rewards]
    return sum(scored) / len(scored) if scored else 0.0


def _is_success(episode: Episode, threshold: float) -> bool:
    return episode.ok and _episode_score(episode) >= threshold


def _trace_timing(trace: Any) -> dict[str, dict[str, float]]:
    agent = trace.timing.agent
    return {
        "agent": {
            "duration": agent.duration,
            "model": agent.model.duration,
            "harness": agent.harness.duration,
        }
    }


def _episode_timing(episode: Episode) -> dict[str, dict[str, float]]:
    """Sum each trace's agent timing to represent an episode's total compute cost."""
    return {
        "agent": {
            "duration": sum(trace.timing.agent.duration for trace in episode.traces),
            "model": sum(trace.timing.agent.model.duration for trace in episode.traces),
            "harness": sum(
                trace.timing.agent.harness.duration for trace in episode.traces
            ),
        }
    }


def _group_brevities(episodes: list[Episode], threshold: float) -> list[float]:
    """Compare total-token lengths only among successful episodes of the same task."""
    successes = [episode for episode in episodes if _is_success(episode, threshold)]
    if not successes:
        return [0.0] * len(episodes)
    lengths = [episode.num_total_tokens for episode in successes]
    longest, shortest = max(lengths), min(lengths)
    return [
        (longest - episode.num_total_tokens) / (longest - shortest + _EPSILON)
        if _is_success(episode, threshold)
        else 0.0
        for episode in episodes
    ]


def _group_time_efficiencies(episodes: list[Episode], threshold: float) -> list[float]:
    """Compare total agent time only among successful episodes of the same task."""
    successes = [episode for episode in episodes if _is_success(episode, threshold)]
    if not successes:
        return [0.0] * len(episodes)
    times = [_episode_timing(episode)["agent"]["duration"] for episode in successes]
    longest, shortest = max(times), min(times)
    return [
        (longest - _episode_timing(episode)["agent"]["duration"])
        / (longest - shortest + _EPSILON)
        if _is_success(episode, threshold)
        else 0.0
        for episode in episodes
    ]


def _group_scores(
    episodes: list[Episode],
    alpha: float,
    threshold: float,
    time_weight: float = 0.5,
) -> list[float]:
    """Score success, relative total-token brevity, and relative agent time."""
    brevities = _group_brevities(episodes, threshold)
    time_efficiencies = _group_time_efficiencies(episodes, threshold)
    return [
        alpha
        + (1 - alpha) * ((1 - time_weight) * brevity + time_weight * time_efficiency)
        if _is_success(episode, threshold)
        else 0.0
        for episode, brevity, time_efficiency in zip(
            episodes, brevities, time_efficiencies, strict=True
        )
    ]


EFFICIENCY_GUIDANCE = (
    "Compare repeated rollouts of the same task under the same parent prompt. "
    "Revise only the evolvable task system prompt. Maximize the weighted score: "
    "correctness 80%, prompt-token efficiency 5%, episode total-token efficiency 7.5%, "
    "and agent-time efficiency 7.5%. Efficiency bonuses apply only to correct rollouts. "
    "Each efficiency term is 1/(1+cost/frozen_seed_scale); compare absolute costs, not "
    "candidate-specific min-max scores. Seek the shortest sufficient prompt. Remove "
    "redundant instructions, task-specific facts, worked examples, and repeated tool "
    "documentation. Add instructions only when expected correctness or execution "
    "savings justify their prompt cost. num_total_tokens measures episode token use, "
    "not final-answer length or provider-billed totals. timing.agent.duration measures "
    "agent seconds; model and harness are diagnostic breakdowns. Learn strategies "
    "that avoid repeated library discovery, redundant tool calls, unnecessary output, "
    "and repeated solution attempts after adequate verification. Preserve checks "
    "needed for feasibility, optimality, and a correct final answer in \\boxed{}. "
    "Use failures to identify missing safeguards, never as examples of efficiency. "
    "Do not infer improvement from timing alone or overfit to one stochastic rollout. "
    "Return a compact, generalizable replacement prompt."
)
