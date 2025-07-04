import cupy
import cupyx

from pubsub import pub
import time

from dataclasses import dataclass
from enum import Enum
import multiprocessing
from multiprocessing.connection import Connection
from typing import Any, cast
import time

import threading
import numpy as np

from src.trackers.tracker_base import (
    TrackerFactoryClassRegistry,
    TrackerFactoryProtocol,
    TrackerProtocol,
)

from cameras.dhyana2100.camera import Camera, CameraConfig
from cameras.camera_protocol import (
    CameraFactoryClassRegistry,
    CameraFactoryProtocol,
    CameraProtocol,
)


BUFFER_SIZE = 300 * 2
NUM_BUFFERS = 5


# IMAGE_HEIGHT = 2016
IMAGE_HEIGHT = 2016 // 2
IMAGE_WIDTH = 2560

CAMERA_ARGUMENTS = {
    "camera_index": 0,
    "enable_fan": False,
    # "number_of_copy_threads_per_buffer": 1,
}

# NUM_ITERS = 1_000
NUM_ITERS = 100


def test_tracker_executor2():
    camera_factory = CameraFactoryClassRegistry.create("dhyana2100", **CAMERA_ARGUMENTS)

    print("CREATING CAMERA")
    camera = camera_factory.create()

    print("Opening camera")
    camera.open()
    camera.set_height(IMAGE_HEIGHT)
    camera.set_width(IMAGE_WIDTH)
    camera.set_framerate(975)

    print(f"WIDTH: {camera.get_width()}")
    print(f"HEIGHT: {camera.get_height()}")
    print(f"FPS:    {camera.get_framerate()}")

    buffers = [
        cupyx.zeros_pinned((BUFFER_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=cupy.uint16)
        for _ in range(NUM_BUFFERS)
    ]

    for buffer in buffers:
        camera.add_buffer(buffer)

    print("START RECORDING")
    camera.start_recording()

    times = []
    for i in range(NUM_ITERS):
        # print(f"{i}/{NUM_ITERS}")
        start = time.perf_counter()
        camera.get_next_buffer()
        end = time.perf_counter()
        times.append(end - start)
        # print(f"Elapsed: {(end - start) * 1_000}ms")

    times = np.array(times)
    print(f"Average: {np.mean(times)}")
    print(f"Std:     {np.std(times)}")

    lost_frames = camera.get_lost_frames()
    print(f"LOST FRAMES BEFORE STOP: {lost_frames}")
    camera.stop_recording()

    lost_frames = camera.get_lost_frames()
    print(f"LOST FRAMES AFTER STOP: {lost_frames}")

    camera.close()

    print("DONE")
