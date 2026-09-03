#!/usr/bin/env python3
"""Generate a privacy-free face video with a known 72 bpm color modulation."""

from pathlib import Path

import cv2
import numpy as np
from skimage import data


def main() -> None:
    destination = Path(__file__).resolve().parent / "videos" / "self_test_72bpm.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    fps, seconds, target_hz = 30.0, 20.0, 1.2
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (image.shape[1], image.shape[0])
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create self-test MP4")
    for frame_index in range(int(fps * seconds)):
        phase = 2.0 * np.pi * target_hz * frame_index / fps
        frame = image.astype(np.float32)
        frame[:, :, 1] *= 1.0 + 0.030 * np.sin(phase)
        frame[:, :, 2] *= 1.0 + 0.010 * np.sin(phase + 0.3)
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()
    print(destination)


if __name__ == "__main__":
    main()
