"""GEPA reflective scoring and timing records."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from verifiers.v1.gepa.adapter import GEPAAdapter, _episode_timing, _group_scores
from verifiers.v1.gepa.config import GEPAConfig


def _episode(
    reward: float,
    total_tokens: int,
    trace_times: list[tuple[float, float]],
    *,
    ok: bool = True,
) -> SimpleNamespace:
    task = SimpleNamespace(data=SimpleNamespace(prompt="question"))
    traces = []
    for index, (model_time, harness_time) in enumerate(trace_times):
        agent_timing = SimpleNamespace(
            duration=model_time + harness_time,
            model=SimpleNamespace(duration=model_time),
            harness=SimpleNamespace(duration=harness_time),
        )
        traces.append(
            SimpleNamespace(
                reward=reward,
                rewards={"task": reward},
                is_dropped=False,
                timing=SimpleNamespace(agent=agent_timing),
                task=task,
                nodes=[],
                agent=SimpleNamespace(name=f"agent-{index}"),
                has_error=False,
                last_error=None,
                stop_condition=None,
                info={},
            )
        )
    return SimpleNamespace(
        ok=ok,
        traces=traces,
        num_total_tokens=total_tokens,
        last_error=None,
        task=task,
    )


def test_episode_timing_sums_agent_model_and_harness_times():
    episode = _episode(1.0, 100, [(2.0, 3.0), (4.0, 1.0)])

    assert _episode_timing(episode) == {
        "agent": {"duration": 10.0, "model": 6.0, "harness": 4.0}
    }


def test_group_scores_mix_relative_brevity_and_agent_time():
    # The shorter rollout is slower, so equal weights give both successful
    # rollouts the midpoint of the success bonus.
    group = [
        _episode(1.0, 100, [(5.0, 15.0)]),
        _episode(1.0, 200, [(4.0, 6.0)]),
        _episode(0.2, 50, [(1.0, 1.0)]),
    ]

    assert _group_scores(group, alpha=0.2, threshold=0.9) == pytest.approx(
        [0.6, 0.6, 0.0]
    )
    length_only = _group_scores(group, alpha=0.2, threshold=0.9, time_weight=0.0)
    time_only = _group_scores(group, alpha=0.2, threshold=0.9, time_weight=1.0)
    assert length_only == pytest.approx([1.0, 0.2, 0.0])
    assert time_only == pytest.approx([0.2, 1.0, 0.0])


def test_group_scores_handle_equal_time_and_no_success():
    equal_times = [
        _episode(1.0, 100, [(4.0, 6.0)]),
        _episode(1.0, 200, [(5.0, 5.0)]),
    ]
    assert _group_scores(equal_times, alpha=0.2, threshold=0.9) == pytest.approx(
        [0.6, 0.2]
    )

    equal_lengths_and_times = [
        _episode(1.0, 100, [(4.0, 6.0)]),
        _episode(1.0, 100, [(5.0, 5.0)]),
    ]
    assert _group_scores(
        equal_lengths_and_times, alpha=0.2, threshold=0.9
    ) == pytest.approx([0.2, 0.2])

    failures = [
        _episode(0.1, 100, [(4.0, 6.0)]),
        _episode(0.2, 200, [(5.0, 5.0)], ok=False),
    ]
    assert _group_scores(failures, alpha=0.2, threshold=0.9) == [0.0, 0.0]


def test_reflective_records_include_trace_and_episode_timing():
    episode = _episode(1.0, 100, [(2.0, 3.0), (4.0, 1.0)])
    adapter = GEPAAdapter(env=None, ctx=None, tasks={}, loop=None)
    eval_batch = SimpleNamespace(trajectories=[[episode]])

    records = adapter.make_reflective_dataset(
        candidate={"system_prompt": "prompt"},
        eval_batch=eval_batch,
        components_to_update=["system_prompt"],
    )["system_prompt"]

    rollout = records[0]["rollouts"][0]
    assert rollout["num_total_tokens"] == 100
    assert "num_output_tokens" not in rollout
    assert rollout["timing"] == {
        "agent": {"duration": 10.0, "model": 6.0, "harness": 4.0}
    }
    assert rollout["traces"][0]["timing"] == {
        "agent": {"duration": 5.0, "model": 2.0, "harness": 3.0}
    }

    standard_records = adapter.make_reflective_dataset(
        candidate={"system_prompt": "prompt"},
        eval_batch=SimpleNamespace(trajectories=[episode]),
        components_to_update=["system_prompt"],
    )["system_prompt"]
    assert standard_records[0]["timing"] == {
        "agent": {"duration": 5.0, "model": 2.0, "harness": 3.0}
    }


class _ImmediateLoop:
    def __init__(self, episodes):
        self.episodes = episodes

    def run_until_complete(self, coroutine):
        coroutine.close()
        return self.episodes


def test_evaluate_uses_group_score_and_keeps_singleton_reward():
    group = [
        _episode(1.0, 100, [(5.0, 15.0)]),
        _episode(1.0, 200, [(4.0, 6.0)]),
        _episode(0.2, 50, [(1.0, 1.0)]),
    ]
    adapter = GEPAAdapter(
        env=None,
        ctx=None,
        tasks={},
        loop=_ImmediateLoop(group),
        train_task_ids={7},
        gr_gepa=True,
        group_size=3,
        group_alpha=0.2,
        group_time_weight=0.5,
        group_success_threshold=0.9,
    )

    assert adapter.evaluate([7], {"system_prompt": "prompt"}).scores == pytest.approx(
        [0.4]
    )

    singleton = _episode(0.7, 100, [(5.0, 15.0)])
    adapter.loop = _ImmediateLoop([singleton])
    assert adapter.evaluate([8], {"system_prompt": "prompt"}).scores == [0.7]


def test_group_time_weight_config_default_and_bounds():
    assert GEPAConfig().group_time_weight == 0.5
    assert (
        GEPAConfig.model_validate({"group_time_weight": 0.0}).group_time_weight == 0.0
    )
    assert (
        GEPAConfig.model_validate({"group_time_weight": 1.0}).group_time_weight == 1.0
    )
    with pytest.raises(ValidationError):
        GEPAConfig.model_validate({"group_time_weight": 1.01})


def _efficient_adapter(**kwargs):
    return GEPAAdapter(
        env=None,
        ctx=None,
        tasks={},
        loop=None,
        efficiency_objective=True,
        count_prompt_tokens=len,
        efficiency_scales={"prompt": 10, "tokens": 100, "agent_seconds": 10},
        **kwargs,
    )


def test_efficiency_rewards_absolute_cost_reductions_and_prompt_compression():
    adapter = _efficient_adapter()
    short = _episode(1, 100, [(5, 5)])
    long = _episode(1, 1000, [(50, 50)])
    assert adapter.efficiency_score(short, "x" * 10) == pytest.approx(0.9)
    assert adapter.efficiency_score(short, "x" * 10) > adapter.efficiency_score(
        long, "x" * 10
    )
    assert adapter.efficiency_score(short, "x") > adapter.efficiency_score(
        short, "x" * 10
    )
    assert adapter.efficiency_score(_episode(0, 1, [(0.1, 0.1)]), "x") == 0
    with pytest.raises(ValueError, match="binary"):
        adapter.efficiency_score(_episode(0.5, 1, [(1, 1)]), "x")


def test_efficiency_calibration_cache_is_counted_once_and_consumed():
    episode = _episode(1, 100, [(5, 5)])
    adapter = _efficient_adapter()
    adapter.loop = _ImmediateLoop([episode])
    adapter.calibrate([8], "x" * 10)
    adapter.loop = None  # Cached evaluation must not call the event loop.
    batch = adapter.evaluate([8], {"system_prompt": "x" * 10}, capture_traces=True)
    assert batch.num_metric_calls == 1
    assert batch.scores == pytest.approx([0.9])
    assert not adapter.calibration_cache
    assert batch.trajectories == [episode]


def test_efficiency_validation_and_group_training_share_scoring():
    episode = _episode(1, 100, [(5, 5)])
    adapter = _efficient_adapter(gr_gepa=True, group_size=2, train_task_ids={7})
    adapter.loop = _ImmediateLoop([episode, episode])
    group = adapter.evaluate([7], {"system_prompt": "x" * 10}, True)
    adapter.loop = _ImmediateLoop([episode])
    single = adapter.evaluate([8], {"system_prompt": "x" * 10}, True)
    assert group.scores == single.scores == pytest.approx([0.9])
    assert group.num_metric_calls == 2
    assert single.num_metric_calls == 1
    records = adapter.make_reflective_dataset(
        {"system_prompt": "x" * 10}, single, ["system_prompt"]
    )["system_prompt"]
    assert records[0]["rollouts"][0]["score"] == pytest.approx(0.9)
    assert records[0]["parent_prompt_tokens"] == 10
    assert adapter.efficiency_scales == {
        "prompt": 10,
        "tokens": 100,
        "agent_seconds": 10,
    }


def test_efficiency_rejects_failed_infrastructure_and_accepts_threshold_one():
    episode = _episode(0, 1, [(1, 1)], ok=False)
    episode.id = "failed"
    adapter = _efficient_adapter()
    with pytest.raises(ValueError, match="failed episode"):
        adapter.efficiency_score(episode, "x")
    assert GEPAConfig(group_success_threshold=1).group_success_threshold == 1
