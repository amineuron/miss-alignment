"""CPU tests for run_iterative_anchoring, focused on the expand_per_step knob.

optimize_fn is mocked (constant loss) so we can count how many full optimizer
solves the schedule triggers and confirm the offset bookkeeping never crashes
or produces NaNs as expand_per_step changes.
"""

import torch

from miss_alignment.alignment.optimize_iterative import run_iterative_anchoring


class _FakeTiltSeries:
    def __init__(self, n_tilts):
        self.n_tilts = n_tilts
        # distinct, finite offsets so chain-restoration is observable
        self.tilt_axis_offset_x = torch.arange(n_tilts, dtype=torch.float32)
        self.tilt_axis_offset_y = torch.arange(n_tilts, dtype=torch.float32) * 10.0

    def indices_sorted_angle(self):
        return torch.arange(self.n_tilts)


def _run(n_tilts, expand_per_step):
    ts = _FakeTiltSeries(n_tilts)
    calls = {"n": 0}

    def optimize_fn(t):
        calls["n"] += 1
        return t, [0.5]  # constant loss; accept/revert path is exercised either way

    out_ts, losses = run_iterative_anchoring(
        ts, optimize_fn, initial_reliable_fraction=0.5, expand_per_step=expand_per_step
    )
    return out_ts, calls["n"]


def test_expand_per_step_1_matches_original_schedule():
    # n=40 -> n_unreliable_per_side=10 -> 10 loop solves + 1 final = 11
    out_ts, n_calls = _run(40, expand_per_step=1)
    assert n_calls == 11
    assert torch.isfinite(out_ts.tilt_axis_offset_x).all()
    assert torch.isfinite(out_ts.tilt_axis_offset_y).all()


def test_expand_per_step_2_runs_fewer_solves():
    # decrement by 2: 10,8,6,4,2 -> 5 loop solves + 1 final = 6
    out_ts, n_calls = _run(40, expand_per_step=2)
    assert n_calls == 6
    assert torch.isfinite(out_ts.tilt_axis_offset_x).all()


def test_expand_per_step_clamped_to_at_least_one():
    # 0 / negative must clamp to 1 (never an infinite loop)
    _, n_zero = _run(40, expand_per_step=0)
    _, n_neg = _run(40, expand_per_step=-5)
    assert n_zero == 11
    assert n_neg == 11


def test_large_expand_collapses_to_two_solves():
    # expand bigger than n_unreliable_per_side: one loop solve + final = 2
    _, n_calls = _run(40, expand_per_step=999)
    assert n_calls == 2
