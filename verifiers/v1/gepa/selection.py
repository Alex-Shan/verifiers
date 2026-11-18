"""Correctness-first selection and equal-opportunity proposal validation."""

from gepa.strategies.eval_policy import FullEvaluationPolicy

from verifiers.v1.gepa.adapter import _episode_score


def measurements(outputs):
    episodes = [
        e for item in outputs for e in (item if isinstance(item, list) else [item])
    ]
    if not episodes:
        raise ValueError("Cannot compare empty evaluations")
    return sum(_episode_score(e) for e in episodes), sum(
        e.num_total_tokens for e in episodes
    ) / len(episodes)


def improves(before, after):
    old_correct, old_tokens = before
    new_correct, new_tokens = after
    return new_correct > old_correct or (
        new_correct == old_correct and new_correct > 0 and new_tokens < old_tokens
    )


class CorrectnessFirstAcceptance:
    def should_accept(self, proposal, state):
        return improves(
            measurements(proposal.eval_before.outputs),
            measurements(proposal.eval_after.outputs),
        )

    def reject_reason(self, proposal, state):
        return f"Correctness-first comparison: {measurements(proposal.eval_before.outputs)} -> {measurements(proposal.eval_after.outputs)}"


class CorrectnessFirstEvaluation(FullEvaluationPolicy):
    def __init__(self, adapter):
        self.adapter = adapter

    def get_best_program(self, state):
        best = 0
        for index in range(1, len(state.program_candidates)):
            before = self.adapter.validation_measurements[
                state.program_candidates[best]["system_prompt"]
            ]
            after = self.adapter.validation_measurements[
                state.program_candidates[index]["system_prompt"]
            ]
            if improves(before, after):
                best = index
        return best


class ValidateAllProposals:
    """Validate rejected proposals too, without adding them to the candidate pool."""

    def __init__(self, adapter, val_ids, path):
        self.adapter, self.val_ids, self.path = adapter, val_ids, path

    def select(self, proposals, state, acceptance_criterion):
        import json

        passing, rejected = [], []
        for proposal in proposals:
            accepted = acceptance_criterion.should_accept(proposal, state)
            (passing if accepted else rejected).append(proposal)
            if self.path:
                with self.path.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "iteration": state.i + 1,
                                "before": measurements(proposal.eval_before.outputs),
                                "after": measurements(proposal.eval_after.outputs),
                                "accepted": accepted,
                                "candidate": proposal.candidate,
                            }
                        )
                        + "\n"
                    )
        for result in self.adapter.batch_evaluate(
            [(p.candidate, self.val_ids) for p in rejected]
        ):
            state.increment_evals(result.num_metric_calls)
        return passing
