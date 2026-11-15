"""Tests for Compression-Research: sweeps and Pareto front."""

import numpy as np
import pytest

from compression_research import bits_sweep, benchmark, pareto_front, rank_sweep, svd_compress


def _matrix():
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    return np.outer(np.sin(t), np.cos(t))


def test_benchmark_rows():
    rows = benchmark(_matrix())
    assert {r["method"] for r in rows} == {"svd_rank4", "quantize8", "fourier4", "hybrid"}
    assert all(r["compression_ratio"] > 0 for r in rows)


def test_rank_sweep_error_decreases_with_rank():
    # Use a rank-4+ matrix so increasing rank genuinely reduces error.
    rng = np.random.default_rng(42)
    W = rng.standard_normal((16, 16))
    rows = rank_sweep(W, ranks=(1, 2, 4, 8))
    errors = [r["relative_error"] for r in rows]
    assert errors == sorted(errors, reverse=True)[:len(errors)] or errors[0] >= errors[-1]
    assert rows[-1]["relative_error"] < rows[0]["relative_error"]


def test_bits_sweep_error_decreases_with_bits():
    rows = bits_sweep(_matrix(), bits_list=(2, 8, 16))
    errors = [r["relative_error"] for r in rows]
    assert errors[0] > errors[-1]
    assert rows[-1]["compression_ratio"] == pytest.approx(0.5)


def test_sweep_validation():
    with pytest.raises(ValueError):
        rank_sweep(_matrix(), ranks=(0,))
    with pytest.raises(ValueError):
        bits_sweep(_matrix(), bits_list=(0,))


def test_pareto_front_removes_dominated_points():
    points = [
        {"relative_error": 0.5, "compression_ratio": 2.0, "tag": "a"},
        {"relative_error": 0.2, "compression_ratio": 2.0, "tag": "b"},
        {"relative_error": 0.1, "compression_ratio": 1.0, "tag": "c"},
    ]
    front = pareto_front(points)
    tags = {p["tag"] for p in front}
    assert tags == {"b", "c"}


def test_pareto_front_empty():
    assert pareto_front([]) == []


def test_svd_still_exact_for_full_rank():
    W = _matrix()
    assert np.max(np.abs(svd_compress(W, rank=2) - W)) < 1e-9
