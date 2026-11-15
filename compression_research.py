# Compression-Research - Rank and bit-depth sweeps with a Pareto front
# Part of the Kapildhinwa2010-dev Toolkit.

"""CompressionResearch - compression trade-off experiments.

Lineage: all toolkit repos. Beyond the fixed benchmark, this module runs
sweeps: SVD rank sweeps, quantization bit-depth sweeps, and a Pareto
front over the (storage, error) trade-off so you can see which operating
points are actually worth choosing.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "svd_compress",
    "quantize",
    "fourier_compress",
    "hybrid_compress",
    "benchmark",
    "rank_sweep",
    "bits_sweep",
    "pareto_front",
]


def _fourier_basis(n_cols: int, num_coeffs: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_cols, endpoint=False)
    cols = [np.ones_like(t)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


def _check(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    return matrix


def svd_compress(matrix: np.ndarray, rank: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if rank < 1:
        raise ValueError("rank must be >= 1")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, s.size)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def quantize(matrix: np.ndarray, bits: int = 8) -> np.ndarray:
    matrix = _check(matrix)
    if bits < 1 or bits > 16:
        raise ValueError("bits must be in [1, 16]")
    lo, hi = float(matrix.min()), float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels)
    return codes * scale + lo


def fourier_compress(matrix: np.ndarray, num_coeffs: int = 4) -> np.ndarray:
    matrix = _check(matrix)
    if num_coeffs < 1:
        raise ValueError("num_coeffs must be >= 1")
    basis = _fourier_basis(matrix.shape[1], num_coeffs)
    coeffs, _, _, _ = np.linalg.lstsq(basis, matrix.T, rcond=None)
    return coeffs.T @ basis.T


def hybrid_compress(matrix: np.ndarray, rank: int = 4, bits: int = 8) -> np.ndarray:
    """SVD low-rank followed by quantization."""
    return quantize(svd_compress(matrix, rank=rank), bits=bits)


def _rel_error(matrix: np.ndarray, approx: np.ndarray) -> float:
    norm = float(np.linalg.norm(matrix))
    return float(np.linalg.norm(matrix - approx)) / norm if norm else 0.0


def benchmark(matrix: np.ndarray) -> list:
    """Run all fixed methods; returns method, error and ratio per row."""
    matrix = _check(matrix)
    methods = {
        "svd_rank4": svd_compress(matrix, rank=4),
        "quantize8": quantize(matrix, bits=8),
        "fourier4": fourier_compress(matrix, num_coeffs=4),
        "hybrid": hybrid_compress(matrix, rank=4, bits=8),
    }
    original = matrix.size * 8
    rows = []
    for name, approx in methods.items():
        rows.append({
            "method": name,
            "relative_error": _rel_error(matrix, approx),
            "compression_ratio": original / max(float(approx.nbytes), 1),
        })
    return rows


def rank_sweep(matrix: np.ndarray, ranks=(1, 2, 4, 8, 16)) -> list:
    """SVD error and ratio for each rank; shows the low-rank trade-off."""
    matrix = _check(matrix)
    rows = []
    m, n = matrix.shape
    for rank in ranks:
        if rank < 1:
            raise ValueError("ranks must be >= 1")
        approx = svd_compress(matrix, rank=rank)
        stored = min(rank, min(m, n)) * (m + n + 1) * 8
        rows.append({
            "rank": int(rank),
            "relative_error": _rel_error(matrix, approx),
            "compression_ratio": matrix.size * 8 / max(stored, 1),
        })
    return rows


def bits_sweep(matrix: np.ndarray, bits_list=(2, 4, 6, 8, 12, 16)) -> list:
    """Quantization error and ratio for each bit depth."""
    matrix = _check(matrix)
    rows = []
    for bits in bits_list:
        if bits < 1 or bits > 16:
            raise ValueError("bits must be in [1, 16]")
        approx = quantize(matrix, bits=bits)
        rows.append({
            "bits": int(bits),
            "relative_error": _rel_error(matrix, approx),
            "compression_ratio": 8.0 / bits,
        })
    return rows


def pareto_front(points: list) -> list:
    """Operating points no other point beats in BOTH error and ratio.

    Each point needs "relative_error" (lower better) and
    "compression_ratio" (higher better). Returns the non-dominated
    subset, sorted by error ascending.
    """
    if not points:
        return []
    front = []
    for p in points:
        dominated = any(
            (q["relative_error"] <= p["relative_error"] and q["compression_ratio"] >= p["compression_ratio"])
            and (q["relative_error"] < p["relative_error"] or q["compression_ratio"] > p["compression_ratio"])
            for q in points if q is not p
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p["relative_error"])
