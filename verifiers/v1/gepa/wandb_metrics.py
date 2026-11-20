"""Log GEPA monitoring metrics to W&B without sending proposal contents."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from gepa.core.callbacks import (
    BudgetUpdatedEvent,
    EvaluationEndEvent,
    IterationEndEvent,
    OptimizationEndEvent,
    ValsetEvaluatedEvent,
)

from verifiers.v1.gepa.config import GEPAConfig

if TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run


class _WandbMetrics:
    def __init__(self, run: Run) -> None:
        self.run = run
        self.best_val_score = float("-inf")
        self.run.define_metric("iteration")
        self.run.define_metric("*", step_metric="iteration")

    def _log(self, iteration: int, **metrics: float) -> None:
        self.run.log({"iteration": iteration, **metrics})

    def on_evaluation_end(self, event: EvaluationEndEvent) -> None:
        scores = event["scores"]
        if scores:
            metric = (
                "train_proposal_score"
                if event["candidate_idx"] is None
                else "train_parent_score"
            )
            self._log(event["iteration"], **{metric: sum(scores) / len(scores)})

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        score = event["average_score"]
        self.best_val_score = max(self.best_val_score, score)
        self._log(
            event["iteration"],
            val_program_average=score,
            best_score_on_valset=self.best_val_score,
            new_program_idx=event["candidate_idx"],
            val_evaluated_count_new_program=event["num_examples_evaluated"],
            val_total_count=event["total_valset_size"],
        )

    def on_budget_updated(self, event: BudgetUpdatedEvent) -> None:
        self._log(event["iteration"], total_metric_calls=event["metric_calls_used"])

    def on_iteration_end(self, event: IterationEndEvent) -> None:
        self._log(
            event["iteration"],
            total_candidates=len(event["state"].program_candidates),
            proposal_accepted=int(event["proposal_accepted"]),
        )

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        self.run.summary.update(
            {
                "best_candidate_idx": event["best_candidate_idx"],
                "best_valset_score": self.best_val_score,
                # GEPA reports the last zero-based iteration index.
                "total_iterations": event["total_iterations"] + 1,
                "total_metric_calls": event["total_metric_calls"],
                "total_candidates": len(event["final_state"].program_candidates),
            }
        )


@contextmanager
def track_wandb_metrics(
    config: GEPAConfig, run_dir: Path | None, num_train: int, num_val: int
) -> Iterator[_WandbMetrics]:
    import wandb

    wandb_dir = run_dir / "wandb" if run_dir is not None else None
    if wandb_dir is not None:
        wandb_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        project=config.wandb_project,
        name=config.run.name,
        dir=str(wandb_dir) if wandb_dir is not None else None,
        settings=wandb.Settings(console="off", disable_code=True, disable_git=True),
        config={
            "model": config.model,
            "reflection_model": config.reflection_model or config.model,
            "num_train": num_train,
            "num_val": num_val,
            "proposal_parents": config.proposal_parents,
            "mutations_per_parent": config.mutations_per_parent,
            "reflection_batch_sampler": config.reflection_batch_sampler,
            "reflection_minibatch_size": config.reflection_minibatch_size,
            "max_total_rollouts": config.max_total_rollouts,
            "max_iterations": config.max_iterations,
            "max_iterations_without_improvement": config.max_iterations_without_improvement,
            "seed": config.seed,
        },
    ) as run:
        yield _WandbMetrics(run)
