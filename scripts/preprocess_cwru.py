#!/usr/bin/env python3
"""Convert CWRU 12 kHz drive-end recordings to the PIAS-SSM format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat


PITCH_DIAMETER_MM = 39.04
BALL_DIAMETER_MM = 7.94
BALL_COUNT = 9
SAMPLING_RATE_HZ = 12_000
RPM_NORMALIZATION = 2000.0
LOAD_TO_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
FILE_NUMBERS = (
    (97, 105, 118, 130, 144, 169, 185, 197, 209, 222),
    (98, 106, 119, 131, 145, 170, 186, 198, 210, 223),
    (99, 107, 120, 132, 146, 171, 187, 199, 211, 224),
    (100, 108, 121, 133, 147, 172, 188, 200, 212, 225),
)


def bearing_coefficients() -> np.ndarray:
    ratio = BALL_DIAMETER_MM / PITCH_DIAMETER_MM
    return np.asarray(
        (
            BALL_COUNT / 2.0 * (1.0 + ratio),
            BALL_COUNT / 2.0 * (1.0 - ratio),
            PITCH_DIAMETER_MM / (2.0 * BALL_DIAMETER_MM) * (1.0 - ratio**2),
            0.5 * (1.0 - ratio),
        ),
        dtype=np.float32,
    )


def resolve_recording(
    data_directory: Path,
    normal_directory: Path | None,
    file_number: int,
    is_normal: bool,
) -> Path:
    candidate = data_directory / f"{file_number}.mat"
    if candidate.is_file():
        return candidate
    if is_normal and normal_directory is not None:
        candidate = normal_directory / f"{file_number}.mat"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing CWRU recording {file_number}.mat")


def read_drive_end(path: Path, file_number: int) -> np.ndarray:
    payload = loadmat(path)
    expected = f"X{file_number:03d}_DE_time"
    if expected in payload:
        return np.asarray(payload[expected]).reshape(-1)
    alternatives = [key for key in payload if "DE_time" in key]
    if not alternatives:
        raise KeyError(f"no drive-end signal in {path}")
    return np.asarray(payload[alternatives[0]]).reshape(-1)


def preprocess(
    data_directory: Path,
    output: Path,
    normal_directory: Path | None,
    window: int,
    stride: int,
    block_size: int,
) -> None:
    if block_size < 1:
        raise ValueError("block_size must be at least one window")
    samples: list[np.ndarray] = []
    labels: list[int] = []
    conditions: list[int] = []
    file_ids: list[int] = []
    rpm_series: list[np.ndarray] = []
    priors: list[np.ndarray] = []
    coefficients = bearing_coefficients()
    source_id = 0

    for condition, row in enumerate(FILE_NUMBERS):
        rpm = LOAD_TO_RPM[condition]
        fault_hz = coefficients * (rpm / 60.0)
        for label, file_number in enumerate(row):
            path = resolve_recording(
                data_directory,
                normal_directory,
                file_number,
                is_normal=label == 0,
            )
            signal = read_drive_end(path, file_number)
            starts = list(range(0, len(signal) - window + 1, stride))
            if len(starts) < 3:
                raise ValueError(f"{path} must yield at least three windows")
            for index, start in enumerate(starts):
                vibration = signal[start : start + window].astype(np.float32)
                vibration = (vibration - vibration.mean()) / (vibration.std() + 1e-8)
                encoded_rpm = np.full(window, rpm / RPM_NORMALIZATION, dtype=np.float32)
                samples.append(np.stack((vibration, encoded_rpm)))
                labels.append(label)
                conditions.append(condition)
                # Consecutive windows are grouped into blocks of `block_size`
                # windows; the trailing remainder forms the last block. Each
                # block receives its own file_id so the leakage-safe split
                # separates whole temporal blocks rather than individual
                # (overlapping-free) windows of the same recording.
                file_ids.append(source_id + index // block_size)
                rpm_series.append(np.full(window, rpm, dtype=np.float32))
                priors.append(np.repeat(fault_hz[:, None], window, axis=1))
            source_id += (len(starts) + block_size - 1) // block_size

    payload = {
        "samples": torch.from_numpy(np.stack(samples)),
        "fault_labels": torch.tensor(labels, dtype=torch.long),
        "condition_labels": torch.tensor(conditions, dtype=torch.long),
        "file_ids": torch.tensor(file_ids, dtype=torch.long),
        "instantaneous_rpm": torch.from_numpy(np.stack(rpm_series)),
        "fault_freq_prior": torch.from_numpy(np.stack(priors)),
        "target_fs": SAMPLING_RATE_HZ,
        "window_len": window,
        "stride": stride,
        "block_size": block_size,
        "split_unit": "non-overlapping temporal block within each source recording",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {output}")
    torch.save(payload, output)
    print(f"saved {len(samples)} samples to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--normal-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=49)
    args = parser.parse_args()
    preprocess(
        args.data_dir,
        args.output,
        args.normal_dir,
        args.window,
        args.stride,
        args.block_size,
    )


if __name__ == "__main__":
    main()
