"""CPU tests for the reconstruction-oversampling env contract.

These lock in the two properties train.py relies on for the per-iteration
oversampling curriculum: (1) the single env source is read consistently, and
(2) re-resolving from a fixed base each iteration means an earlier iteration's
override never leaks into a later one that omits the key.
"""

import os

from miss_alignment.utils import (
    OVERSAMPLING_ENV_VAR,
    reconstruction_oversampling,
)


def test_reads_env_and_falls_back(monkeypatch):
    monkeypatch.delenv(OVERSAMPLING_ENV_VAR, raising=False)
    assert reconstruction_oversampling() == 2.0  # default
    monkeypatch.setenv(OVERSAMPLING_ENV_VAR, "1.5")
    assert reconstruction_oversampling() == 1.5
    monkeypatch.setenv(OVERSAMPLING_ENV_VAR, "not-a-number")
    assert reconstruction_oversampling() == 2.0  # graceful fallback


def test_per_iteration_resolution_no_leak(monkeypatch):
    # Mirror train.py's loop: env = str(iter.get("oversampling", base)) each iter.
    monkeypatch.setenv(OVERSAMPLING_ENV_VAR, "2.0")
    base = os.environ[OVERSAMPLING_ENV_VAR]
    schedule = [
        {"oversampling": 1.25},
        {"oversampling": 1.25},
        {"oversampling": 1.5},
        {},  # omitted -> must fall back to base (2.0), NOT the previous 1.5
        {"oversampling": 2.0, "apply_ctf": True},
    ]
    seen = []
    for s in schedule:
        os.environ[OVERSAMPLING_ENV_VAR] = str(s.get("oversampling", base))
        seen.append(reconstruction_oversampling())
    assert seen == [1.25, 1.25, 1.5, 2.0, 2.0]
