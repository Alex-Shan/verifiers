"""The GEPA <-> v1 bridge: run a candidate system prompt over a batch of tasks and score it.

GEPA's adapter protocol (`evaluate`, `make_reflective_dataset`) is synchronous and
`gepa.api.optimize` blocks, but v1 rollouts are async. The runner manages one event loop by
hand: it enters `env.serving()` on that loop and runs the blocking `optimize()` on the main
thread, so each synchronous `evaluate()` drives its batch of rollouts with
`loop.run_until_complete` — the one sync↔async hop. (Mirrors how v0 vf-gepa bridged; no worker
thread, so a Ctrl-C unwinds straight through `optimize()` into the runner's teardown.)
"""

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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
    semaphore: asyncio.Semaphore | None = None
    on_complete: Callable[[Episode], Awaitable[None]] | None = None
    """Called with each rollout's episode as it finalizes — the runner's persist hook that
    streams episodes to `traces.jsonl`, exactly as `run_eval` does."""
    reflection_columns: list[str] = field(default_factory=list)
    propose_new_texts: Callable[..., Candidate] | None = None
    """Part of GEPA's adapter protocol — its proposer reads this attribute on every reflection
    step. None = use GEPA's default reflection-LM proposer (the AttributeError from leaving it
    undeclared silently disables all mutation proposals)."""
    constraint: str = ""
    reflective_dataset_path: Path | None = None
    gr_gepa: bool = False
    group_size: int = 4
    group_alpha: float = 0.5
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
        n = self.group_size if self.gr_gepa else 1
        episodes = self.loop.run_until_complete(
            self._run_batch(batch, system_prompt, n)
        )
        groups = [episodes[i : i + n] for i in range(0, len(episodes), n)]
        if self.gr_gepa:
            outputs: list[Trajectory] = groups
            scores = [
                sum(
                    _group_scores(group, self.group_alpha, self.group_success_threshold)
                )
                / n
                for group in groups
            ]
        else:
            outputs = episodes
            scores = [_episode_score(episode) for episode in episodes]
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=outputs if capture_traces else None,
            num_metric_calls=len(batch),
        )

    async def _run_batch(
        self, batch: list[int], system_prompt: str, n: int
    ) -> list[Episode]:
        # Inject the candidate as a copy of each base task with its system_prompt overridden
        # (`with_system_prompt` copies rather than reconstructs, so subclass state survives and
        # the shared base task in `self.tasks` is left untouched for the next candidate).
        tasks = [self.tasks[idx].with_system_prompt(system_prompt) for idx in batch]
        slots = [slot for task in tasks for slot in self.env.slots(task, n=n)]
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
            if isinstance(trajectory, list):
                brevities = _group_brevities(trajectory, self.group_success_threshold)
                scores = _group_scores(
                    trajectory, self.group_alpha, self.group_success_threshold
                )
                rollouts = []
                for episode, brevity, score in zip(
                    trajectory, brevities, scores, strict=True
                ):
                    rollouts.append(
                        {
                            "success": int(
                                _is_success(episode, self.group_success_threshold)
                            ),
                            "reward": _episode_score(episode),
                            "num_output_tokens": episode.num_output_tokens,
                            "relative_brevity": brevity,
                            "score": score,
                            "traces": self._episode_records(episode),
                            "error": str(episode.last_error)
                            if episode.last_error
                            else None,
                        }
                    )
                records.append(
                    {
                        "parent_prompt": candidate.get("system_prompt", ""),
                        "query": to_jsonable_python(trajectory[0].task.data.prompt),
                        "comparison_guidance": (
                            "Compare rollouts of this same task. Diagnose recurring errors and "
                            "separate task difficulty from prompt strategy. Reward brevity only "
                            "among successful rollouts; propose a generalizable prompt revision."
                        ),
                        "rollouts": rollouts,
                        "constraint": self.constraint,
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
                "num_output_tokens": trace.num_output_tokens,
                "constraint": self.constraint,
                "agent": trace.agent.name,
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
    signal; an episode with no scored traces scores 0."""
    scored = [trace.reward for trace in episode.traces if trace.rewards]
    return sum(scored) / len(scored) if scored else 0.0


def _is_success(episode: Episode, threshold: float) -> bool:
    return episode.ok and _episode_score(episode) > threshold


def _group_brevities(episodes: list[Episode], threshold: float) -> list[float]:
    """Compare output lengths only among successful episodes of the same task."""
    successes = [episode for episode in episodes if _is_success(episode, threshold)]
    if not successes:
        return [0.0] * len(episodes)
    lengths = [episode.num_output_tokens for episode in successes]
    longest, shortest = max(lengths), min(lengths)
    return [
        (longest - episode.num_output_tokens) / (longest - shortest + _EPSILON)
        if _is_success(episode, threshold)
        else 0.0
        for episode in episodes
    ]


def _group_scores(
    episodes: list[Episode], alpha: float, threshold: float
) -> list[float]:
    """Apply the success/relative-brevity tradeoff to each rollout in a group."""
    return [
        alpha + (1 - alpha) * brevity if _is_success(episode, threshold) else 0.0
        for episode, brevity in zip(
            episodes, _group_brevities(episodes, threshold), strict=True
        )
    ]
