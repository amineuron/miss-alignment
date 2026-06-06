# Speed-vs-quality scenarios

A matrix of `iteration_settings` you can run and compare. Everything here is
**opt-in**: with no per-iteration keys set you get the *original* behaviour, so
the `orig` scenario below is your "without these changes" baseline.

## Two independent groups of knobs

1. **Reconstruction-fidelity curriculum** (what these scenarios vary):
   per-iteration `downsample`, `oversampling`, `apply_ctf`. These change *both*
   training and alignment within an iteration.
2. **Alignment-only speed knobs** (orthogonal — kept OFF here so the comparison
   isolates group 1): `lbfgs_options`, `anchoring_expand_per_step`,
   `foreground_keep_fraction` in the `tilt_series_alignment` block. Add them to
   any scenario later to speed the alignment phase further; A/B them separately.

> Quick reminder (see chat): `oversampling 1.25` is the aggressive part, and it
> is *riskier at ds1 (full res)* than at ds3 — at ds3 there's little high-freq to
> alias. If unsure, floor at 1.5.

## The matrix (alignment modes held fixed = anchoring,anchoring,global,[3,3]×5)

| Scenario | downsample | oversampling | CTF | Purpose |
|---|---|---|---|---|
| `orig` | 3,2,1,1,1,1,1,1 | (2.0 default) | no | **baseline = no changes** |
| `orig_ctf` | 3,2,1,1,1,1,1,1 | 2.0 | iters 7–8 | CTF effect alone |
| `os_curve` | **1,1,1,1,1,1,1,1** | 1.25,1.25,1.5,1.5,1.75,1.75,2,2 | no | oversampling effect, full-res (⚠ anchoring at ds1 is slow) |
| `os_curve_ctf` | 1×8 | same | iters 7–8 | + CTF |
| `both_curve` | **3,3,2,2,1,1,1,1** | 1.25,1.25,1.5,1.5,1.75,1.75,2,2 | no | downsample + oversampling curriculum |
| `both_curve_ctf` | 3,3,2,2,1,1,1,1 | same | iters 7–8 | the "fast + quality" candidate |

## iteration_settings blocks

`orig` (baseline — what you get with no changes):
```yaml
  iteration_settings:
    - { downsample: 3, alignment: anchoring }
    - { downsample: 2, alignment: anchoring }
    - { downsample: 1, alignment: global }
    - { downsample: 1, alignment: [3, 3] }
    - { downsample: 1, alignment: [3, 3] }
    - { downsample: 1, alignment: [3, 3] }
    - { downsample: 1, alignment: [3, 3] }
    - { downsample: 1, alignment: [3, 3] }
```

`orig_ctf` — as `orig` but add `, apply_ctf: true` to the last two rows.

`os_curve` (downsample 1 always; ⚠ anchoring at full res on iters 1–2 is expensive):
```yaml
  iteration_settings:
    - { downsample: 1, alignment: anchoring, oversampling: 1.25 }
    - { downsample: 1, alignment: anchoring, oversampling: 1.25 }
    - { downsample: 1, alignment: global,    oversampling: 1.5 }
    - { downsample: 1, alignment: [3, 3],    oversampling: 1.5 }
    - { downsample: 1, alignment: [3, 3],    oversampling: 1.75 }
    - { downsample: 1, alignment: [3, 3],    oversampling: 1.75 }
    - { downsample: 1, alignment: [3, 3],    oversampling: 2.0 }
    - { downsample: 1, alignment: [3, 3],    oversampling: 2.0 }
```

`both_curve` (downsample curriculum + oversampling curriculum):
```yaml
  iteration_settings:
    - { downsample: 3, alignment: anchoring, oversampling: 1.25 }
    - { downsample: 3, alignment: anchoring, oversampling: 1.25 }
    - { downsample: 2, alignment: global,    oversampling: 1.5 }
    - { downsample: 2, alignment: [3, 3],    oversampling: 1.5 }
    - { downsample: 1, alignment: [3, 3],    oversampling: 1.75 }
    - { downsample: 1, alignment: [3, 3],    oversampling: 1.75 }
    - { downsample: 1, alignment: [3, 3],    oversampling: 2.0 }
    - { downsample: 1, alignment: [3, 3],    oversampling: 2.0 }
```

`os_curve_ctf` / `both_curve_ctf` — add `, apply_ctf: true` to rows 7–8.

## A 100%-pristine baseline (no C3 / no logging either)

The `orig` scenario above is functionally identical to the original code — every
new feature is off by default — *except* the always-on inductor cache (C3, faster
recompile, **identical results**) and one extra log line. For a literally
untouched baseline, run the `main` branch (or your production install) instead;
results will match `orig`, only the per-iteration recompile time differs by a few
seconds.

## How to run

Each scenario is a separate run into its own `--output_dir` (or, standalone,
`general.training_directory`). Either:
- **RELION**: launch an AlignTiltSeries job per scenario and paste the block into
  the generated `missalign_config.yaml` before it trains (or set
  `MISS_RECONSTRUCTION_OVERSAMPLING` in the environment for a flat override), or
- **standalone**: copy a working `missalign_config.yaml`, swap in the block, point
  `training_directory` at a fresh copy of the prepared data, and run
  `miss-alignment --config-file scenario.yaml ...`.

Time each run (wall clock) — that's your speed axis.

## How to compare (quality axis)

Alignment **loss is not comparable across scenarios** (each trains its own
scorer). Compare the final *shifts* against the trusted `orig` baseline:

```bash
python compare_alignments.py \
    --ref  orig:/path/orig/external/training \
    --run  both_curve:/path/both_curve/external/training \
    --run  both_curve_ctf:/path/both_curve_ctf/external/training
```

A fast scenario whose mean RMS Δshift from `orig` is small (≈ sub-pixel) is "as
good, but faster". For CTF scenarios a deviation is expected and may be an
*improvement* — confirm with a downstream STA resolution when it matters.
