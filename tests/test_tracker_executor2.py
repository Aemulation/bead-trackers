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

IMAGE_TOPIC = "TrackerExecutorImageTopic"

# TODO: Only used for testing. Remove later and make configurable.
BUFFER_SIZE = 300


class TrackerExecutorWorkerCommand(Enum):
    Start = 1
    Stop = 2


class TrackerExecutorControllerCommand(Enum):
    Image = 1


class TrackerExecutorWorker:
    def __init__(
        self,
        camera_factory: CameraFactoryProtocol,
        tracker_factory: TrackerFactoryProtocol,
        roi_coordinates: np.ndarray,
    ):
        self.__camera_factory = camera_factory
        self.__tracker_factory = tracker_factory
        self.__roi_coordinates = roi_coordinates

    def __setup_tracking(
        self,
        buffer_size: int,
        num_host_frame_buffers: int,
        roi_coordinates: np.ndarray,
    ):
        num_rois = roi_coordinates.shape[0]

        self.__camera = self.__camera_factory.create()

        self.__camera.open()
        frame_height = self.__camera.get_height()
        frame_width = self.__camera.get_width()
        frame_size = self.__camera.get_frame_size()
        bytes_per_pixel = frame_size // frame_height // frame_width
        array_type = TrackerExecutorWorker.array_type(bytes_per_pixel)

        self.__host_frame_buffers = [
            cupyx.zeros_pinned(
                (
                    buffer_size,
                    frame_height,
                    frame_width,
                ),
                dtype=array_type,
            )
            for _ in range(num_host_frame_buffers)
        ]
        for buffer in self.__host_frame_buffers:
            self.__camera.add_buffer(buffer)

        self.__device_frame_buffer1 = cupy.empty(
            (buffer_size, frame_height, frame_width), dtype=array_type
        )
        self.__device_frame_buffer2 = cupy.empty(
            (buffer_size, frame_height, frame_width), dtype=array_type
        )

        self.__tracker1 = self.__tracker_factory.create()
        self.__tracker2 = self.__tracker_factory.create()

        self.__stream1 = cupy.cuda.Stream(non_blocking=True)
        self.__stream2 = cupy.cuda.Stream(non_blocking=True)

        self.host_z_values_buffer1 = cupyx.zeros_pinned(
            (buffer_size, num_rois), dtype=cupy.float32
        )
        self.host_z_values_buffer2 = cupyx.zeros_pinned(
            (buffer_size, num_rois), dtype=cupy.float32
        )

        # Warmup. Compile all Cupy and CUDA code.
        for _ in range(5):
            self.__tracker1.calculate(self.__device_frame_buffer1)
        for _ in range(5):
            self.__tracker2.calculate(self.__device_frame_buffer2)

    def __teardown_tracking(self):
        self.__camera.close()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    @staticmethod
    def array_type(bytes_per_pixel: int) -> np.typing.DTypeLike:
        if bytes_per_pixel == 1:
            return cupy.uint8
        if bytes_per_pixel == 2:
            return cupy.uint16

    def __run_tracker(self):
        self.__camera.start_recording()
        host_images_buffer1 = self.__camera.get_next_buffer()
        cupy.cuda.runtime.memcpyAsync(
            self.__device_frame_buffer1.data.ptr,
            host_images_buffer1.ctypes.data,
            host_images_buffer1.nbytes,
            cupy.cuda.runtime.memcpyHostToDevice,
            self.__stream1.ptr,
        )

        while True:
            if self.__running_lock.acquire(blocking=False):
                running = self.__running
                self.__running_lock.release()
                if not running:
                    break

            # TODO: Wait for transfer completion using events.
            # TODO: Copy z values to host
            # TODO: Ignore first z values
            with self.__stream2:
                self.__tracker1.calculate(self.__device_frame_buffer1)
                device_z_values_buffer1 = self.__tracker1.get_calculated_z()

            self.__controller_pipe.send(
                (TrackerExecutorControllerCommand.Image, host_images_buffer1[0])
            )
            host_images_buffer2 = self.__camera.get_next_buffer()
            cupy.cuda.runtime.memcpyAsync(
                self.__device_frame_buffer2.data.ptr,
                host_images_buffer2.ctypes.data,
                host_images_buffer2.nbytes,
                cupy.cuda.runtime.memcpyHostToDevice,
                self.__stream1.ptr,
            )

            # TODO: Wait for transfer completion using events.
            # TODO: Copy z values to host
            with self.__stream2:
                self.__tracker2.calculate(self.__device_frame_buffer2)
                device_z_values_buffer2 = self.__tracker2.get_calculated_z()

            self.__controller_pipe.send(
                (TrackerExecutorControllerCommand.Image, host_images_buffer2[0])
            )
            host_images_buffer1 = self.__camera.get_next_buffer()
            cupy.cuda.runtime.memcpyAsync(
                self.__device_frame_buffer1.data.ptr,
                host_images_buffer1.ctypes.data,
                host_images_buffer1.nbytes,
                cupy.cuda.runtime.memcpyHostToDevice,
                self.__stream1.ptr,
            )

        self.__camera.stop_recording()

    def run(
        self,
        controller_pipe: Connection,
    ):
        self.__controller_pipe = controller_pipe

        self.__running = False
        self.__running_lock = threading.Lock()

        # TODO: Make configurable
        self.__setup_tracking(BUFFER_SIZE, 5, self.__roi_coordinates)

        self.__tracker_thread = threading.Thread(target=self.__run_tracker)

        self.__run_communication()

        self.__teardown_tracking()

    def __run_communication(self):
        while True:
            while not self.__controller_pipe.poll():
                time.sleep(0.1)
            command = self.__controller_pipe.recv()

            if command == TrackerExecutorWorkerCommand.Start:
                self.__start()
            elif command == TrackerExecutorWorkerCommand.Stop:
                self.__stop()
                return
            else:
                print(f"Received unknown command: {command}")

    def __start(self):
        with self.__running_lock:
            self.__running = True
        self.__tracker_thread.start()

    def __stop(self):
        with self.__running_lock:
            self.__running = False

        self.__tracker_thread.join()

        # TODO: Remove
        print(f"LOST FRAMES: {self.__camera.get_lost_frames()}")
        self.__camera.close()


class TrackerExecutorController:
    def __init__(
        self,
        camera_factory: CameraFactoryProtocol,
        tracker_factory: TrackerFactoryProtocol,
        roi_coordinates: np.ndarray,
    ):
        self.__controller_to_worker_pipe, self.__worker_to_controller_pipe = (
            multiprocessing.Pipe()
        )

        self.__worker = TrackerExecutorWorker(
            camera_factory, tracker_factory, roi_coordinates
        )

        self.__running = False
        self.__running_lock = threading.Lock()

        self.__communication_thread = threading.Thread(
            target=self.__run_communication, args=()
        )

        multiprocessing.set_start_method("spawn")
        self.__worker_process = multiprocessing.Process(
            target=self.__worker.run, args=(self.__worker_to_controller_pipe,)
        )

    def __run_communication(self):
        while True:
            if not self.__controller_to_worker_pipe.poll(timeout=0.1):
                if not self.__running:
                    break
                time.sleep(0.1)
                continue

            (command, data) = self.__controller_to_worker_pipe.recv()
            print(f"Controller got command: {command}")
            if command == TrackerExecutorControllerCommand.Image:
                image = cast(np.ndarray, data)
                pub.sendMessage(IMAGE_TOPIC, image=image)

    def start(self):
        with self.__running_lock:
            self.__running = True

        self.__communication_thread.start()
        self.__worker_process.start()
        self.__controller_to_worker_pipe.send(TrackerExecutorWorkerCommand.Start)

    def stop(self):
        with self.__running_lock:
            self.__running = False

        self.__controller_to_worker_pipe.send(TrackerExecutorWorkerCommand.Stop)
        self.__worker_process.join()
        self.__communication_thread.join()


NUM_ROIS = 10
ROI_COORDINATES = cupy.array([[0, 0]] * NUM_ROIS, dtype=cupy.uint32)
ROI_SIZE = 100

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


def test_tracker_executor2():
    tracker_factory = TrackerFactoryClassRegistry.create(
        "qi_tracker", **QI_TRACKER_ARGUMENTS
    )
    camera_factory = CameraFactoryClassRegistry.create("dhyana2100")

    tracker_executor = TrackerExecutorController(
        camera_factory, tracker_factory, ROI_COORDINATES
    )

    print("STARTING TRACKER EXECUTOR")
    tracker_executor.start()

    print("SLEEPING")
    time.sleep(30)
    print("WOKE-UP")

    print("STOPPING TRACKER EXECUTOR")
    tracker_executor.stop()

    print("DONE")
