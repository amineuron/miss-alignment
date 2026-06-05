"""CPU tests for foreground position masking (select_foreground_positions).

Reconstruction is mocked so each patch's variance is a strictly increasing
function of its position index, making the kept subset deterministic.
"""

import torch

from miss_alignment.alignment.tilt_series import select_foreground_positions


class _FakeTiltSeries:
    """Minimal stand-in: no-op .to() and a reconstruct that fakes variance.

    Patch j is `base * (coords[j, 0] + 1)`, so var(patch_j) grows monotonically
    with the position's first coordinate -> higher coordinate == kept first.
    """

    def to(self, device):  # noqa: D401 - mirrors warpylib TiltSeries.to
        return self

    @staticmethod
    def reconstruct_subvolumes_single(**kwargs):
        coords = kwargs["coords"]
        size = kwargs["size"]
        base = torch.arange(size**3, dtype=torch.float32).reshape(size, size, size)
        scales = coords[:, 0] + 1.0
        return base[None] * scales[:, None, None, None]


def _positions(n):
    return torch.stack([torch.tensor([float(i), 0.0, 0.0]) for i in range(n)])


def test_keep_fraction_one_is_noop():
    positions = _positions(10)
    out = select_foreground_positions(
        _FakeTiltSeries(),
        images=torch.zeros(1),
        pixel_size=1.0,
        positions=positions,
        patch_size=4,
        keep_fraction=1.0,
        device="cpu",
    )
    assert out is positions  # untouched, same object


def test_keeps_highest_variance_positions():
    positions = _positions(10)
    out = select_foreground_positions(
        _FakeTiltSeries(),
        images=torch.zeros(1),
        pixel_size=1.0,
        positions=positions,
        patch_size=4,
        keep_fraction=0.5,
        batch_size=3,  # exercise multi-batch indexing (3,3,3,1)
        device="cpu",
    )
    assert out.shape[0] == 5
    # the five highest-variance positions are indices 5..9
    assert sorted(out[:, 0].tolist()) == [5.0, 6.0, 7.0, 8.0, 9.0]
    # order is preserved (ascending index), not shuffled by topk
    assert out[:, 0].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_rounds_up_to_at_least_one():
    positions = _positions(3)
    out = select_foreground_positions(
        _FakeTiltSeries(),
        images=torch.zeros(1),
        pixel_size=1.0,
        positions=positions,
        patch_size=4,
        keep_fraction=0.01,  # rounds to 0 -> clamped to 1
        device="cpu",
    )
    assert out.shape[0] == 1
    assert out[0, 0].item() == 2.0  # the single highest-variance position
