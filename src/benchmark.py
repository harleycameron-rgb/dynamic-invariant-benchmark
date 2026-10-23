#!/usr/bin/env python3
"""Compare dense, sparse, and phase-plus-POD representations of a dynamic field."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TRIALS = 5
BITS_PER_VALUE = 8
SPATIAL_BITS = 10


def generate_field(field: str, grid: int, frames: int, seed: int) -> np.ndarray:
    """Return a frames-by-pixels field for the selected case."""
    rng = np.random.default_rng(seed)
    if field == "noise":
        return rng.random((frames, grid * grid), dtype=np.float64)

    axis = np.linspace(-1.0, 1.0, grid)
    xx, yy = np.meshgrid(axis, axis)
    phase_offset = rng.uniform(0.0, 2.0 * np.pi)
    blobs = (
        (0.58, 0.00, 0.90, 0.12),
        (0.38, 1.15, 0.68, 0.16),
        (0.65, 2.50, 0.48, 0.13),
        (0.46, 4.10, 0.35, 0.19),
        (0.28, 5.30, 0.25, 0.12),
    )
    amplitude_scale = 0.86
    width_scale = 1.7

    snapshots = []
    for index in range(frames):
        angle = phase_offset + 2.0 * np.pi * index / frames
        image = np.full((grid, grid), 0.12, dtype=np.float64)
        for radius, initial_angle, amplitude, width in blobs:
            center_angle = initial_angle + angle
            cx = radius * np.cos(center_angle)
            cy = radius * np.sin(center_angle)
            image += amplitude * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2)
                / (2.0 * (width * width_scale) ** 2)
            )
        image = 0.12 + (image - 0.12) * amplitude_scale
        snapshots.append(image.ravel())
    return np.asarray(snapshots)


def quantize(values: np.ndarray, bits: int = BITS_PER_VALUE) -> np.ndarray:
    """Uniformly quantize values over their observed range."""
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low:
        return np.full_like(values, low)
    levels = (1 << bits) - 1
    encoded = np.rint((values - low) * levels / (high - low))
    return low + encoded * (high - low) / levels


def quantize_components(values: np.ndarray, bits: int = BITS_PER_VALUE) -> np.ndarray:
    """Quantize each row independently, preserving small POD components."""
    return np.asarray([quantize(row, bits) for row in values])


def evaluate_trial(field: str, grid: int, frames: int, modes: int, seed: int) -> dict:
    data = generate_field(field, grid, frames, seed)
    mean = data.mean(axis=0)
    centered = data - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    basis = right_vectors[:modes]
    coefficients = centered @ basis.T

    energy = float(np.sum(singular_values[:modes] ** 2) / np.sum(singular_values**2))
    oracle = mean + coefficients @ basis
    oracle_mse = float(np.mean((data - oracle) ** 2))

    encoded_mean = quantize(mean, SPATIAL_BITS)
    encoded_basis = np.asarray([quantize(row, SPATIAL_BITS) for row in basis])
    encoded_coefficients = quantize_components(coefficients.T, SPATIAL_BITS).T
    reconstruction = encoded_mean + encoded_coefficients @ encoded_basis
    invariant_mse = float(np.mean((data - reconstruction) ** 2))

    raw_frames = quantize_components(data)
    raw_mse = float(np.mean((data - raw_frames) ** 2))

    total_bits = (
        (modes + 1) * grid * grid * SPATIAL_BITS
        + frames * modes * SPATIAL_BITS
        + frames * BITS_PER_VALUE
    )
    samples_per_frame = min(
        grid * grid,
        max(1, (total_bits - BITS_PER_VALUE) // (frames * BITS_PER_VALUE)),
    )
    sample_rng = np.random.default_rng(seed ^ 0x5A17)
    sample_indices = sample_rng.choice(grid * grid, samples_per_frame, replace=False)
    mean_fill = quantize(np.asarray([data.mean()]))[0]
    sparse = np.full_like(data, mean_fill)
    sparse[:, sample_indices] = data[:, sample_indices]
    sparse_mse = float(np.mean((data - sparse) ** 2))
    sparse_bits = int(samples_per_frame * frames * BITS_PER_VALUE + BITS_PER_VALUE)

    return {
        "pod_energy_captured": energy,
        "phase_plus_modes_mse": invariant_mse,
        "pod_oracle_float64_mse": oracle_mse,
        "raw_frames_mse": raw_mse,
        "raw_at_bit_budget_mse": sparse_mse,
        "bit_budget": int(total_bits),
        "raw_at_bit_budget_bits": sparse_bits,
        "sparse_samples_per_frame": int(samples_per_frame),
        "wins": invariant_mse < sparse_mse,
        "sentinel": hashlib.sha512(
            np.asarray(data, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
    }


def make_summary(field: str, grid: int, frames: int, modes: int, seed: int) -> dict:
    trials = [
        evaluate_trial(field, grid, frames, modes, seed + trial)
        for trial in range(TRIALS)
    ]
    average = lambda key: float(np.mean([trial[key] for trial in trials]))
    return {
        "schema_version": 1,
        "field": field,
        "config": {
            "grid": grid,
            "frames": frames,
            "modes": modes,
            "seed": seed,
            "trials": TRIALS,
        },
        "metrics": {
            "pod_energy_captured": average("pod_energy_captured"),
            "phase_plus_modes_mse": average("phase_plus_modes_mse"),
            "pod_oracle_float64_mse": average("pod_oracle_float64_mse"),
            "raw_frames_mse": average("raw_frames_mse"),
            "raw_at_bit_budget_mse": average("raw_at_bit_budget_mse"),
            "bit_budget": trials[0]["bit_budget"],
            "raw_at_bit_budget_bits": trials[0]["raw_at_bit_budget_bits"],
            "sparse_samples_per_frame": trials[0]["sparse_samples_per_frame"],
        },
        "win_counts": {
            "phase_plus_modes_vs_raw_at_bit_budget": {
                "wins": sum(trial["wins"] for trial in trials),
                "trials": TRIALS,
            }
        },
        "sha512_sentinel": trials[0]["sentinel"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=32, help="square grid width (default: 32)")
    parser.add_argument("--frames", type=int, default=48, help="number of snapshots (default: 48)")
    parser.add_argument("--modes", type=int, default=4, help="number of POD modes (default: 4)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--field", choices=("designed", "noise"), default="designed",
        help="designed rotating blobs or i.i.d. noise control",
    )
    parser.add_argument("--out", type=Path, help="summary JSON output path")
    args = parser.parse_args()
    if args.grid < 2 or args.frames < 2 or args.modes < 1 or args.modes >= args.frames:
        parser.error("grid and frames must be >= 2; modes must be in [1, frames)")
    return args


def main() -> None:
    args = parse_args()
    summary = make_summary(args.field, args.grid, args.frames, args.modes, args.seed)
    output = args.out or (
        Path(__file__).resolve().parent.parent
        / "results"
        / f"summary_{args.field}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
