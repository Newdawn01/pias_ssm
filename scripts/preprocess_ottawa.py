#!/usr/bin/env python3
"""Convert University of Ottawa recordings to the PIAS-SSM format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import medfilt, resample_poly


ORIGINAL_RATE_HZ = 200_000
TARGET_RATE_HZ = 2_000
ENCODER_PULSES_PER_REVOLUTION = 1024
RPM_NORMALIZATION = 2000.0
FAULTS = {
    "H": (0, "1 Data collected from a healthy bearing"),
    "I": (1, "2 Data collected from a bearing with inner race fault"),
    "O": (2, "3 Data collected from a bearing with outer race fault"),
    "B": (3, "4 Data collected from a bearing with ball fault"),
    "C": (4, "5 Data collected from a bearing with a combination of faults"),
}
CONDITIONS = {"A": 0, "B": 1, "C": 2, "D": 3}


def bearing_coefficients() -> np.ndarray:
    pitch, ball, count = 38.52, 7.94, 9
    ratio = ball / pitch
    return np.asarray(
        (
            count / 2.0 * (1.0 + ratio),
            count / 2.0 * (1.0 - ratio),
            pitch / (2.0 * ball) * (1.0 - ratio**2),
            0.5 * (1.0 - ratio),
        ),
        dtype=np.float32,
    )


def resolve_folder(raw_directory: Path, folder: str) -> Path:
    """Return the fault folder, tolerating minor name differences.

    The published archive names the ball-fault directory ``4 Data collected
    from ball fault``, while some mirrors write ``a ball fault``. Matching on
    the leading index (``4``) removes that ambiguity.
    """

    direct = raw_directory / folder
    if direct.is_dir():
        return direct
    prefix = folder.split()[0]
    matches = [
        candidate
        for candidate in raw_directory.iterdir()
        if candidate.is_dir() and candidate.name.split()[0] == prefix
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"cannot locate folder '{folder}' under {raw_directory}"
    )


def instantaneous_rpm(encoder: np.ndarray) -> np.ndarray:
    rising = np.where(np.diff((encoder > 2.0).astype(np.int8)) == 1)[0]
    if rising.size < 2:
        return np.full(encoder.size, 1200.0, dtype=np.float32)
    intervals = np.diff(rising).astype(np.float64)
    rpm_values = ORIGINAL_RATE_HZ / intervals / ENCODER_PULSES_PER_REVOLUTION * 60.0
    rpm_values = medfilt(rpm_values, kernel_size=5)
    output = np.empty(encoder.size, dtype=np.float64)
    output[: rising[0]] = rpm_values[0]
    for index in range(rising.size - 1):
        output[rising[index] : rising[index + 1]] = rpm_values[index]
    output[rising[-1] :] = rpm_values[-1]
    return output.astype(np.float32)


def process_recording(
    path: Path,
    fault_label: int,
    condition_label: int,
    file_id: int,
    window: int,
    stride: int,
) -> list[dict[str, np.ndarray | int]]:
    payload = loadmat(path)
    vibration = np.asarray(payload["Channel_1"]).reshape(-1).astype(np.float64)
    encoder = np.asarray(payload["Channel_2"]).reshape(-1).astype(np.float64)
    vibration -= vibration.mean()
    rpm = instantaneous_rpm(encoder)
    vibration = resample_poly(vibration, TARGET_RATE_HZ, ORIGINAL_RATE_HZ).astype(np.float32)
    rpm = resample_poly(rpm, TARGET_RATE_HZ, ORIGINAL_RATE_HZ).astype(np.float32)
    length = min(vibration.size, rpm.size)
    vibration, rpm = vibration[:length], rpm[:length]
    priors = bearing_coefficients()[:, None] * (rpm[None, :] / 60.0)
    records: list[dict[str, np.ndarray | int]] = []
    for start in range(0, length - window + 1, stride):
        stop = start + window
        segment = vibration[start:stop]
        segment = (segment - segment.mean()) / (segment.std() + 1e-8)
        records.append(
            {
                "sample": np.stack(
                    (segment, rpm[start:stop] / RPM_NORMALIZATION)
                ).astype(np.float32),
                "rpm": rpm[start:stop].copy(),
                "prior": priors[:, start:stop].astype(np.float32),
                "label": fault_label,
                "condition": condition_label,
                "file_id": file_id,
            }
        )
    return records


def preprocess(raw_directory: Path, output: Path, window: int, stride: int) -> None:
    records: list[dict[str, np.ndarray | int]] = []
    file_map: dict[int, str] = {}
    file_id = 0
    for fault_code, (fault_label, folder) in FAULTS.items():
        fault_directory = resolve_folder(raw_directory, folder)
        for condition_code, condition_label in CONDITIONS.items():
            for repetition in (1, 2, 3):
                filename = f"{fault_code}-{condition_code}-{repetition}.mat"
                path = fault_directory / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                records.extend(
                    process_recording(
                        path,
                        fault_label,
                        condition_label,
                        file_id,
                        window,
                        stride,
                    )
                )
                file_map[file_id] = filename
                file_id += 1
    payload = {
        "samples": torch.from_numpy(np.stack([record["sample"] for record in records])),
        "fault_labels": torch.tensor([record["label"] for record in records]),
        "condition_labels": torch.tensor([record["condition"] for record in records]),
        "file_ids": torch.tensor([record["file_id"] for record in records]),
        "instantaneous_rpm": torch.from_numpy(np.stack([record["rpm"] for record in records])),
        "fault_freq_prior": torch.from_numpy(np.stack([record["prior"] for record in records])),
        "file_id_map": file_map,
        "target_fs": TARGET_RATE_HZ,
        "window_len": window,
        "stride": stride,
        "split_unit": "source recording",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {output}")
    torch.save(payload, output)
    print(f"saved {len(records)} samples from {file_id} files to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=128)
    args = parser.parse_args()
    preprocess(args.raw_dir, args.output, args.window, args.stride)


if __name__ == "__main__":
    main()
