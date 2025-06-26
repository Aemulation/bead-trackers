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


# TODO: Only used for testing. Remove later and make configurable.
BUFFER_SIZE = 300


class TrackerExecutorWorkerCommand(Enum):
    Start = 1
    Stop = 2


class TrackerExecutorControllerCommand(Enum):
    Image = 1
    ZCoordinates = 2
    YXCoordinates = 3


class TrackerExecutorWorker:
    def __init__(
        self,
        camera_factory: CameraFactoryProtocol,
        tracker_factory: TrackerFactoryProtocol,
        roi_coordinates: np.ndarray,
        frame_height: int,
        frame_width: int,
        framerate: int,
    ):
        self.__camera_factory = camera_factory
        self.__tracker_factory = tracker_factory
        self.__roi_coordinates = roi_coordinates

        self.__frame_height = frame_height
        self.__frame_width = frame_width
        self.__framerate = framerate

    def __setup_tracking(
        self,
        buffer_size: int,
        num_host_frame_buffers: int,
        roi_coordinates: np.ndarray,
    ):
        num_rois = roi_coordinates.shape[0]

        self.__camera = self.__camera_factory.create()

        self.__camera.open()
        self.__camera.set_height(self.__frame_height)
        self.__camera.set_width(self.__frame_width)
        self.__camera.set_framerate(self.__framerate)

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

        self.__host_z_values_buffer1 = cupyx.zeros_pinned(
            (buffer_size, num_rois), dtype=cupy.float32
        )
        self.__host_z_values_buffer2 = cupyx.zeros_pinned(
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
        transfer_frames_done_event1 = cupy.cuda.Event(disable_timing=True)
        transfer_frames_done_event2 = cupy.cuda.Event(disable_timing=True)
        transfer_coordinates_done_event1 = cupy.cuda.Event(disable_timing=True)
        transfer_coordinates_done_event2 = cupy.cuda.Event(disable_timing=True)

        self.__camera.start_recording()

        host_images_buffer1 = self.__camera.get_next_buffer()
        self.__stream1.synchronize()
        self.__stream2.synchronize()
        with self.__stream1:
            cupy.cuda.runtime.memcpyAsync(
                self.__device_frame_buffer1.data.ptr,
                host_images_buffer1.ctypes.data,
                host_images_buffer1.nbytes,
                cupy.cuda.runtime.memcpyHostToDevice,
                self.__stream1.ptr,
            )
            transfer_frames_done_event1.record()

        # Grab some dummy data
        # TODO: Can we expand this pre-loop code so it contains valid data?
        device_z_values_buffer2 = self.__tracker2.get_calculated_z()

        while self.__keep_running.is_set():
            print("Tracker starting while loop")
            with self.__stream2:
                self.__stream2.wait_event(transfer_frames_done_event2)
                transfer_frames_done_event2 = cupy.cuda.Event(disable_timing=True)
                self.__tracker1.calculate(self.__device_frame_buffer1)
                device_z_values_buffer1 = self.__tracker1.get_calculated_z()

            self.__controller_pipe.send(
                (TrackerExecutorControllerCommand.Image, host_images_buffer1[0])
            )
            self.__controller_pipe.send(
                (
                    TrackerExecutorControllerCommand.ZCoordinates,
                    device_z_values_buffer2,
                )
            )
            # TODO: Send y,x and z coordinates from buffer 2 to controller.
            self.__stream1.wait_event(transfer_coordinates_done_event1)
            transfer_coordinates_done_event1 = cupy.cuda.Event(disable_timing=True)

            host_images_buffer2 = self.__camera.get_next_buffer()
            self.__stream1.synchronize()
            self.__stream2.synchronize()
            with self.__stream1:
                cupy.cuda.runtime.memcpyAsync(
                    self.__device_frame_buffer2.data.ptr,
                    host_images_buffer2.ctypes.data,
                    host_images_buffer2.nbytes,
                    cupy.cuda.runtime.memcpyHostToDevice,
                    self.__stream1.ptr,
                )
                transfer_frames_done_event2.record()
                cupy.cuda.runtime.memcpyAsync(
                    self.__host_z_values_buffer1.ctypes.data,
                    device_z_values_buffer1.data.ptr,
                    device_z_values_buffer1.nbytes,
                    cupy.cuda.runtime.memcpyDeviceToHost,
                    self.__stream1.ptr,
                )
                transfer_coordinates_done_event1.record()

            with self.__stream2:
                self.__stream2.wait_event(transfer_frames_done_event1)
                transfer_frames_done_event1 = cupy.cuda.Event(disable_timing=True)
                self.__tracker2.calculate(self.__device_frame_buffer2)
                device_z_values_buffer2 = self.__tracker2.get_calculated_z()

            self.__controller_pipe.send(
                (TrackerExecutorControllerCommand.Image, host_images_buffer2[0])
            )
            self.__controller_pipe.send(
                (TrackerExecutorControllerCommand.ZCoordinates, device_z_values_buffer1)
            )
            # TODO: Send y,x and z coordinates to controller.
            self.__stream1.wait_event(transfer_coordinates_done_event2)
            transfer_coordinates_done_event2 = cupy.cuda.Event(disable_timing=True)

            host_images_buffer1 = self.__camera.get_next_buffer()
            self.__stream1.synchronize()
            self.__stream2.synchronize()
            with self.__stream1:
                cupy.cuda.runtime.memcpyAsync(
                    self.__device_frame_buffer1.data.ptr,
                    host_images_buffer1.ctypes.data,
                    host_images_buffer1.nbytes,
                    cupy.cuda.runtime.memcpyHostToDevice,
                    self.__stream1.ptr,
                )
                transfer_frames_done_event1.record()
                cupy.cuda.runtime.memcpyAsync(
                    self.__host_z_values_buffer2.ctypes.data,
                    device_z_values_buffer2.data.ptr,
                    device_z_values_buffer2.nbytes,
                    cupy.cuda.runtime.memcpyDeviceToHost,
                    self.__stream1.ptr,
                )
                transfer_coordinates_done_event2.record()
            print("Tracker reached end of for loop")

        print("Worker stopping camera")
        self.__camera.stop_recording()
        print("Worker done tracking")
        self.__tracker_done.set()

    def run(
        self,
        controller_pipe: multiprocessing.Queue,
    ):
        self.__controller_pipe = controller_pipe
        self.__keep_running = threading.Event()
        self.__tracker_done = threading.Event()

        # TODO: Make configurable
        self.__setup_tracking(BUFFER_SIZE, 5, self.__roi_coordinates)

        self.__run_communication()

        self.__teardown_tracking()

    def __run_communication(self):
        while True:
            while self.__controller_pipe.empty():
                time.sleep(0.1)
            command = self.__controller_pipe.get()

            if command == TrackerExecutorWorkerCommand.Start:
                print("Worker starting...")
                self.__start()
            elif command == TrackerExecutorWorkerCommand.Stop:
                print("Worker stopping...")
                self.__stop()
                return
            else:
                print(f"Worker received unsupported command: {command}")

    def __start(self):
        if self.__keep_running.is_set():
            return
        self.__keep_running.set()
        self.__tracker_done.clear()

        self.__tracker_thread = threading.Thread(target=self.__run_tracker)
        self.__tracker_thread.start()

    def __stop(self):
        if not self.__keep_running.is_set():
            return
        self.__keep_running.clear()

        print("Waiting for tracker to be done")
        self.__tracker_done.wait()
        print("Worker joining tracker thread")
        self.__tracker_thread.join()

        # TODO: Remove
        print(f"LOST FRAMES: {self.__camera.get_lost_frames()}")


class TrackerExecutorController:
    IMAGE_TOPIC = "TrackerExecutorImageTopic"
    Z_VALUES_TOPIC = "TrackerExecutorZValuesTopic"

    def __init__(
        self,
        camera_factory: CameraFactoryProtocol,
        tracker_factory: TrackerFactoryProtocol,
        roi_coordinates: np.ndarray,
    ):
        self.__controller_to_worker_queue = multiprocessing.Queue()
        self.__worker_to_controller_queue = multiprocessing.Queue()

        self.__worker = TrackerExecutorWorker(
            camera_factory,
            tracker_factory,
            roi_coordinates,
            frame_height=2016,
            frame_width=2560,
            framerate=970,
        )

        self.__running = False
        self.__running_lock = threading.Lock()

        self.__communication_thread = threading.Thread(
            target=self.__run_communication, args=()
        )

        # multiprocessing.set_start_method("spawn")
        self.__worker_process = multiprocessing.Process(
            target=self.__worker.run, args=(self.__worker_to_controller_queue,)
        )

    def __run_communication(self):
        while True:
            if self.__controller_to_worker_queue.empty():
                if not self.__running:
                    break
                time.sleep(0.1)
                continue

            (command, data) = self.__controller_to_worker_queue.get()
            print(f"Controller got command: {command}")
            if command == TrackerExecutorControllerCommand.Image:
                image = cast(np.ndarray, data)
                pub.sendMessage(self.IMAGE_TOPIC, image=image)
            elif command == TrackerExecutorControllerCommand.ZCoordinates:
                image = cast(np.ndarray, data)
                pub.sendMessage(self.Z_VALUES_TOPIC, image=image)
            else:
                print(f"Controller received unsupported command: {command}")
        print("Controller communication is done")

    def start(self):
        with self.__running_lock:
            self.__running = True

        self.__communication_thread.start()
        self.__worker_process.start()
        self.__controller_to_worker_queue.put(TrackerExecutorWorkerCommand.Start)

    def stop(self):
        self.__controller_to_worker_queue.put(TrackerExecutorWorkerCommand.Stop)
        self.__worker_process.join()

        with self.__running_lock:
            self.__running = False
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

CAMERA_ARGUMENTS = {
    "camera_index": 0,
    "enable_fan": False,
    "number_of_copy_threads_per_buffer": 1,
}


def test_tracker_executor2():
    tracker_factory = TrackerFactoryClassRegistry.create(
        "qi_tracker", **QI_TRACKER_ARGUMENTS
    )
    camera_factory = CameraFactoryClassRegistry.create("dhyana2100", **CAMERA_ARGUMENTS)

    tracker_executor = TrackerExecutorController(
        camera_factory, tracker_factory, ROI_COORDINATES
    )

    print("STARTING TRACKER EXECUTOR")
    tracker_executor.start()

    print("SLEEPING")
    time.sleep(90)
    print("WOKE-UP")

    print("STOPPING TRACKER EXECUTOR")
    tracker_executor.stop()

    print("DONE")
