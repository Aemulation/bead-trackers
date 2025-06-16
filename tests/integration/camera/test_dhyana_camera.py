import cupy
import cupyx
import time
import pytest
import threading

import time

from src.trackers.bead_tracker.tracker import Tracker
from src.trackers.bead_tracker.radial_profiler import RadialProfilerConfig
from src.trackers.tracker_base import TrackerProtocol

from cameras.dhyana2100.camera import Camera, CameraConfig
from cameras.camera_protocol import CameraFactoryClassRegistry, CameraProtocol


ROI_SIZE = 100
NUM_Z_LAYERS = 100
NUM_RADIALS = ROI_SIZE // 4
NUM_ANGLE_STEPS = 100


def radial_profiler_config():
    return RadialProfilerConfig(1, ROI_SIZE / 4, NUM_RADIALS, NUM_ANGLE_STEPS)


@pytest.fixture
def mock_z_lookup_table():
    return cupy.repeat(
        cupy.expand_dims(
            cupy.linspace(10, 20, NUM_Z_LAYERS, dtype=cupy.float32), axis=0
        ).T,
        NUM_RADIALS,
        axis=1,
    )


@pytest.fixture
def mock_z_values():
    return cupy.linspace(100, 1100, NUM_Z_LAYERS, dtype=cupy.float32)


def run_test(
    streams,
    device_images_buffers,
    host_z_values_buffers,
    trackers: list[TrackerProtocol],
    camera,
    num_rounds,
):
    for i in range(num_rounds):
        print(f"{i}/{num_rounds}")
        for (
            stream,
            device_images_buffer,
            host_z_values_buffer,
            tracker,
        ) in zip(
            streams,
            device_images_buffers,
            host_z_values_buffers,
            trackers,
        ):
            host_images_buffer = camera.get_next_buffer()

            with stream:
                stream.synchronize()

                cupy.cuda.runtime.memcpyAsync(
                    device_images_buffer.data.ptr,
                    host_images_buffer.ctypes.data,
                    host_images_buffer.nbytes,
                    cupy.cuda.runtime.memcpyHostToDevice,
                    stream.ptr,
                )

                tracker.calculate(device_images_buffer)
                z_values = tracker.get_calculated_z()

                cupy.cuda.runtime.memcpyAsync(
                    host_z_values_buffer.ctypes.data,
                    z_values.data.ptr,
                    z_values.nbytes,
                    cupy.cuda.runtime.memcpyDeviceToHost,
                    stream.ptr,
                )


def run_test2(
    streams,
    device_images_buffers,
    host_z_values_buffers,
    trackers: list[TrackerProtocol],
    camera: CameraProtocol,
    num_rounds,
):
    [stream1, stream2] = streams
    [device_images_buffer1, device_images_buffer2] = device_images_buffers
    [host_z_values_buffer1, host_z_values_buffer2] = host_z_values_buffers
    # [tracker1, tracker2] = trackers
    [tracker] = trackers

    def handle_result(host_z_values_buffer):
        print("GOT Z VALUES")

    for i in range(num_rounds):
        # print(f"{i}/{num_rounds}, dropped frames: {camera.get_lost_frames()}")

        start = time.perf_counter()

        host_images_buffer1 = camera.get_next_buffer()
        end = time.perf_counter()
        print(f"getting next buffer took {end - start}")

        cupy.cuda.runtime.memcpyAsync(
            device_images_buffer1.data.ptr,
            host_images_buffer1.ctypes.data,
            host_images_buffer1.nbytes,
            cupy.cuda.runtime.memcpyHostToDevice,
            stream1.ptr,
        )
        # print(f"stream2: {stream2.done}")
        # stream2.synchronize()
        # start = time.perf_counter()
        with stream2:
            tracker.calculate(device_images_buffer2)
            device_z_values_buffer2 = tracker.get_calculated_z()

        # end = time.perf_counter()
        # print(f"tracker1 took {end - start}")
        device_z_values_buffer2.get(
            stream=stream2, out=host_z_values_buffer2, blocking=False
        )
        stream2.launch_host_func(handle_result, host_z_values_buffer2)

        start = time.perf_counter()
        host_images_buffer2 = camera.get_next_buffer()
        end = time.perf_counter()
        print(f"getting next buffer took {end - start}")

        cupy.cuda.runtime.memcpyAsync(
            device_images_buffer2.data.ptr,
            host_images_buffer2.ctypes.data,
            host_images_buffer2.nbytes,
            cupy.cuda.runtime.memcpyHostToDevice,
            stream2.ptr,
        )
        # print(f"stream1: {stream1.done}")
        # stream1.synchronize()
        # start = time.perf_counter()
        with stream1:
            tracker.calculate(device_images_buffer1)
            device_z_values_buffer1 = tracker.get_calculated_z()
        # end = time.perf_counter()
        # print(f"tracker2 took {end - start}")
        device_z_values_buffer1.get(
            stream=stream1, out=host_z_values_buffer1, blocking=False
        )
        stream1.launch_host_func(handle_result, host_z_values_buffer1)


def make_roi_coordinates(
    num_rois: int, image_height: int, image_width: int, roi_size: int
) -> cupy.ndarray:
    random_y = cupy.random.randint(
        0, image_height - roi_size, num_rois, dtype=cupy.uint32
    )
    random_x = cupy.random.randint(
        0, image_width - roi_size, num_rois, dtype=cupy.uint32
    )

    return cupy.sort(cupy.column_stack((random_y, random_x)))


def test_dropped_frames(mock_z_lookup_table: cupy.ndarray, mock_z_values: cupy.ndarray):
    print("\nHello from cupy-test!")

    # mock_bead_coordinates = cupy.array(
    #     [
    #         [387, 1380],
    #         [774, 817],
    #         [965, 455],
    #         [1468, 128],
    #     ],
    #     dtype=cupy.uint32,
    # )
    # mock_bead_coordinates = (
    #     cupy.repeat(mock_bead_coordinates, 100, axis=0).astype(cupy.uint32).copy()
    # )
    # roi_coordinates = (
    #     mock_bead_coordinates
    #     - cupy.array([roi_height // 2, roi_width // 2], dtype=cupy.uint32).copy()
    # )
    num_rois = 600

    z_lookup_tables = cupy.array([mock_z_lookup_table for _ in range(num_rois)])

    num_images = 300

    print("CREATING CAMERA")
    dhyana2100_arguments = {
        "camera_index": 0,
        "enable_fan": False,
        "number_of_copy_threads_per_buffer": 64,
    }
    camera = CameraFactoryClassRegistry.create("dhyana2100", **dhyana2100_arguments)
    print("OPENING CAMERA")
    camera.open()

    camera.set_width(2560)
    camera.set_height(2016)
    camera.set_framerate(970)

    print("CAMERA CONFIG")
    print(camera.get_width())
    print(camera.get_height())
    print(camera.get_framerate())

    image_height = camera.get_height()
    image_width = camera.get_width()
    print(f"FPS: {camera.get_framerate()}")

    print(f"image_height: {image_height}")
    print(f"image_width: {image_width}")

    roi_coordinates = make_roi_coordinates(
        num_rois, image_height, image_width, ROI_SIZE
    )
    num_host_buffers = 5
    host_images_buffers = [
        cupyx.zeros_pinned((num_images, image_height, image_width), dtype=cupy.uint16)
        for _ in range(num_host_buffers)
    ]
    num_streams = 2
    device_images_buffers = [
        cupy.empty((num_images, image_height, image_width), dtype=cupy.uint16)
        for _ in range(num_streams)
    ]
    streams = [cupy.cuda.Stream(non_blocking=True) for _ in range(num_streams)]
    host_z_values_buffers = [
        cupyx.zeros_pinned((num_images, num_rois), dtype=cupy.float32)
        for _ in range(num_streams)
    ]

    for host_buffer in host_images_buffers:
        print("ADDING BUFFER")
        camera.add_buffer(host_buffer)

    print("CREATING TRACKER")
    lookup_table_images = cupy.zeros((num_rois, 100, ROI_SIZE, ROI_SIZE))
    trackers: list[TrackerProtocol] = [
        Tracker(
            num_images_per_buffer=num_images,
            roi_coordinates=roi_coordinates,
            roi_size=ROI_SIZE,
            lookup_table_images=lookup_table_images,
            min_qi_radius=1,
            max_qi_radius=ROI_SIZE / 4,
            number_of_qi_radial_steps=NUM_RADIALS,
            number_of_qi_angle_steps=NUM_ANGLE_STEPS,
            number_of_qi_iterations=3,
            min_lut_radius=1,
            max_lut_radius=ROI_SIZE / 4,
            number_of_lut_radial_steps=NUM_RADIALS,
            number_of_lut_angle_steps=NUM_ANGLE_STEPS,
        )
        # for _ in range(num_streams)
        for _ in range(1)
    ]
    #
    print("DRY RUN TRACKER")
    # Dry run to compile all cupy code.
    for tracker in trackers:
        for _ in range(100):
            tracker.calculate(device_images_buffers[0])

    # time.sleep(3)
    # print(f"Dropped frames before: {camera.get_lost_frames()}")
    print("START RECORDING")
    start = time.perf_counter()
    camera.start_recording()

    print("DRY RUN CAMERA + TRACKER")
    num_dry_run_rounds = 10
    run_test2(
        streams,
        device_images_buffers,
        host_z_values_buffers,
        trackers,
        camera,
        num_dry_run_rounds,
    )
    print(f"Dropped after warm-up: {camera.get_lost_frames()}")

    num_rounds = 3000
    run_test2(
        streams,
        device_images_buffers,
        host_z_values_buffers,
        trackers,
        camera,
        num_rounds,
    )

    print(f"Dropped before stop: {camera.get_lost_frames()}")
    print("TEMPERATURE")
    camera.stop_recording()
    print(f"Dropped after stop: {camera.get_lost_frames()}")
    print("TEMPERATURE")
    for stream in streams:
        stream.synchronize()
        # np.append(computed_z_values, computed_z_values1)

    end = time.perf_counter()
    print(f"Dropped frames after: {camera.get_lost_frames()}")
    print(f"Elapsed: {end - start}")
    assert camera.get_lost_frames() == 0
