import cupy
import cupyx

from pubsub import pub

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


BUFFER_SIZE = 300
NUM_BUFFERS = 5


NUM_ROIS = 10
ROI_COORDINATES = cupy.array([[0, 0]] * NUM_ROIS, dtype=cupy.uint32)
ROI_SIZE = 100

IMAGE_HEIGHT = 2016
IMAGE_WIDTH = 2560

QI_TRACKER_ARGUMENTS = {
    "num_images_per_buffer": BUFFER_SIZE,
    "roi_coordinates": ROI_COORDINATES,
    "roi_size": ROI_SIZE,
    "zstacks": cupy.zeros([NUM_ROIS, 20, ROI_SIZE, ROI_SIZE], dtype=cupy.uint16),
    "number_of_qi_radial_steps": ROI_SIZE // 4,
    "number_of_qi_angle_steps": 100,
    "number_of_lut_radial_steps": 100,
    "number_of_lut_angle_steps": 100,
}

CAMERA_ARGUMENTS = {
    "camera_index": 0,
    "enable_fan": False,
    "number_of_copy_threads_per_buffer": 1,
}

# NUM_ITERS = 1_000
NUM_ITERS = 100


def test_tracker_executor2():
    tracker_factory = TrackerFactoryClassRegistry.create(
        "qi_tracker", **QI_TRACKER_ARGUMENTS
    )
    camera_factory = CameraFactoryClassRegistry.create("dhyana2100", **CAMERA_ARGUMENTS)

    print("CREATING CAMERA")
    camera = camera_factory.create()

    print("Opening camera")
    camera.open()
    camera.set_height(IMAGE_HEIGHT)
    camera.set_height(IMAGE_WIDTH)
    camera.set_framerate(975)

    buffers = [
        np.zeros((BUFFER_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint16)
        for _ in range(NUM_BUFFERS)
    ]

    for buffer in buffers:
        camera.add_buffer(buffer)

    print("START RECORDING")
    camera.start_recording()

    for i in range(NUM_ITERS):
        print(f"{i}/{NUM_ITERS}")
        camera.get_next_buffer()

    lost_frames = camera.get_lost_frames()
    print(f"LOST FRAMES BEFORE STOP: {lost_frames}")
    camera.stop_recording()

    lost_frames = camera.get_lost_frames()
    print(f"LOST FRAMES AFTER STOP: {lost_frames}")

    print("DONE")
