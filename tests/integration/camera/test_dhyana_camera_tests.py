import cupy
import cupyx
import time
import pytest
import json

from src.tracker import Tracker
from src.radial_profiler import RadialProfilerConfig

from dhyana_camera import Camera, CameraConfig

NUM_Z_LAYERS = 100
NUM_RADIALS = 25
NUM_ANGLE_STEPS = 100

ROI_SIZE = 100


def make_radial_profiler_config():
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
    trackers,
    camera,
    num_rounds,
) -> tuple[cupy.ndarray, cupy.ndarray]:
    transfer_to_device_times = []
    transfer_to_host_times = []
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

                e1 = cupy.cuda.Event()
                e1.record()
                cupy.cuda.runtime.memcpy(
                    device_images_buffer.data.ptr,
                    host_images_buffer.ctypes.data,
                    host_images_buffer.nbytes,
                    cupy.cuda.runtime.memcpyHostToDevice,
                )
                e2 = stream.record()
                stream.wait_event(e2)
                e2.synchronize()
                t = cupy.cuda.get_elapsed_time(e1, e2)
                transfer_to_device_times.append(t)

                (yx_coordinates, z_values) = tracker.compute_z_values(
                    device_images_buffer
                )

                e1 = cupy.cuda.Event()
                e1.record()
                cupy.cuda.runtime.memcpy(
                    host_z_values_buffer.ctypes.data,
                    z_values.data.ptr,
                    z_values.nbytes,
                    cupy.cuda.runtime.memcpyDeviceToHost,
                )
                e2 = stream.record()
                stream.wait_event(e2)
                e2.synchronize()
                t = cupy.cuda.get_elapsed_time(e1, e2)
                transfer_to_host_times.append(t)

    return transfer_to_device_times, transfer_to_host_times


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

    num_rois = 300

    z_lookup_tables = cupy.array([mock_z_lookup_table for _ in range(num_rois)])

    buffer_sizes = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for num_images in buffer_sizes:
        print("CREATING CAMERA")
        camera = Camera()
        print("OPENING CAMERA")
        camera.open(0, 128)

        print("DISABLEING FAN")
        camera.set_temperature(False, True)

        print("TEMPERATURE")
        print(camera.get_temperature())

        print("UPDATING CONFIG")
        camera.update_configs()

        print("CAMERA CONFIG")
        print(camera.config.to_dict())

        print("SETTING CONFIG")
        camera_config = CameraConfig(
            **{
                "width": 2560,
                "height": 2016,
                "offsetx": 0,
                "offsety": 0,
                "exposure_ms": 1,
                "framerate": 970,
                "num_buffer_frames": 256,
                "fast_binning": 1,
                "trigger_frames": 1,
                "bits_per_pixel": 12,
                "input_trigger": 0,
                "full_mode": 1,
            }
        )
        camera.set_config(camera_config)

        print("CAMERA CONFIG")
        print(camera.config.to_dict())

        print("SETTING CONFIG")
        camera_config = CameraConfig(
            **(
                camera.config.to_dict()
                | {
                    "framerate": 970,
                }
            )
        )

        print("CAMERA CONFIG")
        print(camera.config.to_dict())

        image_height = camera.config.height
        image_width = camera.config.width
        print(f"FPS: {camera.config.framerate}")

        print(f"image_height: {image_height}")
        print(f"image_width: {image_width}")

        roi_coordinates = make_roi_coordinates(
            num_rois, image_height, image_width, ROI_SIZE
        )
        num_host_buffers = 3
        host_images_buffers = [
            cupyx.zeros_pinned(
                (num_images, image_height, image_width), dtype=cupy.uint16
            )
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
        radial_profiler_config = make_radial_profiler_config()
        trackers = [
            Tracker(
                z_lookup_tables,
                mock_z_values,
                roi_coordinates,
                ROI_SIZE,
                ROI_SIZE,
                num_images,
                radial_profiler_config,
            )
            for _ in range(num_streams)
        ]

        print("DRY RUN TRACKER")
        # Dry run to compile all cupy code.
        for tracker in trackers:
            for _ in range(100):
                tracker.compute_z_values(device_images_buffers[0])

        time.sleep(3)
        print(f"Dropped frames before: {camera.get_lost_frames()}")
        start = time.perf_counter()
        camera.start()

        print("DRY RUN CAMERA + TRACKER")
        num_dry_run_rounds = 10
        for _ in range(num_dry_run_rounds):
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
                    (yx_coordinates, z_values) = tracker.compute_z_values(
                        device_images_buffer
                    )
                    cupy.cuda.runtime.memcpyAsync(
                        host_z_values_buffer.ctypes.data,
                        z_values.data.ptr,
                        z_values.nbytes,
                        cupy.cuda.runtime.memcpyDeviceToHost,
                        stream.ptr,
                    )
        print(f"Dropped after warm-up: {camera.get_lost_frames()}")

        data = {}
        num_rounds = 1_000 // num_streams
        (transfer_to_device_times, transfer_to_host_times) = run_test(
            streams,
            device_images_buffers,
            host_z_values_buffers,
            trackers,
            camera,
            num_rounds,
        )
        data["parameters"] = {
            "num_rois": num_rois,
            "roi_size": ROI_SIZE,
            "max_radial": radial_profiler_config.max_radius,
            "num_radial_steps": radial_profiler_config.num_radial_steps,
            "num_angle_steps": radial_profiler_config.num_angle_steps,
            "buffer_size": num_images,
        }
        data["transfer_to_host_times"] = transfer_to_host_times
        data["transfer_to_device_times"] = transfer_to_device_times

        file_name = ",".join(
            [f"{key}={value}" for key, value in data["parameters"].items()]
        )

        with open(f"test-results/data-transfer/{file_name}.json", "w") as file:
            json.dump(data, file)

        print(f"Dropped before stop: {camera.get_lost_frames()}")
        print("TEMPERATURE")
        print(camera.get_temperature())
        camera.stop()
        print(f"Dropped after stop: {camera.get_lost_frames()}")
        print("TEMPERATURE")
        camera.get_temperature()
        for stream in streams:
            stream.synchronize()
            # np.append(computed_z_values, computed_z_values1)

        end = time.perf_counter()
        print(f"Dropped frames after: {camera.get_lost_frames()}")
        print(f"Elapsed: {end - start}")
        camera.close()
