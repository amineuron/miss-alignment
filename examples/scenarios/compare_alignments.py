"""Compare final tilt-series alignments across scenario runs.

IMPORTANT: alignment *loss* values are NOT comparable across scenarios. Each run
trains its own scorer, so the score scale differs run-to-run -- a lower loss in
one run does not mean a better alignment than another run. The meaningful
cross-scenario metric is how far each run's final per-tilt shifts deviate from a
trusted *reference* run (e.g. the all-oversampling-2.0 baseline). A faster
scenario that lands on almost the same shifts as the reference is "as good, but
faster". (The gold standard remains a downstream STA resolution; this is the
cheap proxy you can compute immediately from the output XMLs.)

Usage
-----
    python compare_alignments.py \\
        --ref   orig:/path/AlignTiltSeries/orig/external/training \\
        --run   os_curve:/path/AlignTiltSeries/os_curve/external/training \\
        --run   both_curve:/path/AlignTiltSeries/both_curve/external/training

Each value is `label:training_dir`, where training_dir holds the final aligned
`*.xml` files (the per-tilt-series Warp XMLs the run wrote). Reports, per
scenario, the mean / max RMS per-tilt 2D shift difference (Angstrom) from the
reference, over the tilt-series present in both.

Note on CTF scenarios: a deviation from a no-CTF reference is expected (and may
be an *improvement*) -- this metric measures "how different", not "how worse".
Read it together with each run's wall-clock time.
"""

import argparse
from pathlib import Path

import torch
from warpylib import TiltSeries


def load_offsets(training_dir: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return {tilt_series_name: (offset_x, offset_y)} for the *.xml in a dir."""
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for xml in sorted(Path(training_dir).glob("*.xml")):
        ts = TiltSeries(xml)
        out[xml.stem] = (
            ts.tilt_axis_offset_x.detach().cpu().float(),
            ts.tilt_axis_offset_y.detach().cpu().float(),
        )
    return out


def rms_shift_diff(
    a: tuple[torch.Tensor, torch.Tensor],
    b: tuple[torch.Tensor, torch.Tensor],
) -> float:
    """RMS of the per-tilt 2D shift difference (Angstrom)."""
    dx, dy = a[0] - b[0], a[1] - b[1]
    return torch.sqrt((dx**2 + dy**2).mean()).item()


def _parse_spec(spec: str) -> tuple[str, str]:
    label, _, path = spec.partition(":")
    if not path:
        raise SystemExit(f"Expected label:path, got {spec!r}")
    return label, path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="label:training_dir of the reference")
    ap.add_argument(
        "--run", action="append", default=[], help="label:training_dir (repeatable)"
    )
    args = ap.parse_args()

    ref_label, ref_dir = _parse_spec(args.ref)
    ref = load_offsets(ref_dir)
    print(f"Reference: {ref_label}  ({len(ref)} tilt-series)\n")

    header = f"{'scenario':<16}{'mean RMS Δshift (Å)':>22}{'max (Å)':>12}{'n':>5}"
    print(header)
    print("-" * len(header))
    for spec in args.run:
        label, d = _parse_spec(spec)
        runs = load_offsets(d)
        common = sorted(set(ref) & set(runs))
        if not common:
            print(f"{label:<16}{'(no matching tilt-series)':>22}")
            continue
        diffs = torch.tensor([rms_shift_diff(ref[n], runs[n]) for n in common])
        print(f"{label:<16}{diffs.mean().item():>22.3f}{diffs.max().item():>12.3f}"
              f"{len(common):>5}")


if __name__ == "__main__":
    main()
