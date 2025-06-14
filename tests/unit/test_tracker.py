import cupy
import numpy as np
import pytest
import json

from src.trackers.bead_tracker.quadrant_interpolation_tracker import (
    QuadrantInterpolationTracker,
)
from src.trackers.bead_tracker.tracker import Tracker
from src.trackers.bead_tracker.radial_profiler import (
    RadialProfiler,
    RadialProfilerConfig,
)

import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.animation

from tests.unit.conftest import NUM_RADIALS, NUM_Z_LAYERS, make_roi_coordinates

import time


NUM_IMAGES = 10
ROI_SIZE = 60

NUM_RADIAL_STEPS = (ROI_SIZE // 4) - 1
NUM_ANGLE_STEPS = 131

REFERENCE_BEAD_ID = 96

Z_CORRECTION = 0.88


# DATA_DIRECTORY = (
#     "/home/markhonkoop/thesis/RNA67_hairpin_force_extension_save_ROI_tracker"
# )
DATA_DIRECTORY = (
    "C:/data/Misha/20250520/exp1/RNA67_hairpin_force_extension_save_ROI_tracker"
)
# DATA_DIRECTORY = "C:/data/Luca/20250414/TX_mtRNAP_7pN"
# DATA_DIRECTORY = "C:/data/Misha/20250520/exp1/RNA67_hairpin_const_forces_save_ROI"
# DATA_DIRECTORY = "/home/markhonkoop/tmp/RNA67_hairpin_force_extension_save_ROI_tracker"

BEADS_TO_VISUALIEZ = np.array([0, 11, 20, 31, 40, 50, 54, -1])
# BEADS_TO_VISUALIEZ = np.array([0, 1, 2, 3, 4, 5, 6, 7])
# BEADS_TO_VISUALIEZ = np.array([8, 9, 10, 11, 12, 13, 14, 15])
# BEADS_TO_VISUALIEZ = np.array([16, 17, 18, 19, 20, 21, 22, 23])


@pytest.fixture
def zstacks() -> cupy.ndarray:
    return cupy.load(f"{DATA_DIRECTORY}/zstacks.npy")


@pytest.fixture
def roi_coordinates() -> cupy.ndarray:
    bead_coordinates = cupy.loadtxt(f"{DATA_DIRECTORY}/beadpos_xy.txt", dtype=int)
    roi_coordinates = bead_coordinates - cupy.array([ROI_SIZE // 2, ROI_SIZE // 2])
    roi_coordinates[:, [0, 1]] = roi_coordinates[:, [1, 0]]

    return roi_coordinates.astype(cupy.uint32)


class RoiGetter:
    def __init__(self) -> None:
        self.__counter = 0
        self.__file = open(f"{DATA_DIRECTORY}/rois.npy", "rb")

    def __reset(self):
        self.__file.close()
        self.__file = open(f"{DATA_DIRECTORY}/rois.npy", "rb")
        self.__counter = 0

    def __del__(self):
        self.__file.close()

    def get_roi_np(self, timestamp: int, roi_id: int | np.ndarray) -> np.ndarray:
        if self.__counter > timestamp:
            self.__reset()

        while True:
            try:
                rois = np.load(self.__file)
            except Exception:
                self.__reset()
                rois = np.load(self.__file)

            if self.__counter == timestamp:
                return rois[roi_id]

            self.__counter += 1


def get_roi_np(timestamp: int, roi_id: int | np.ndarray) -> np.ndarray:
    with open(f"{DATA_DIRECTORY}/rois.npy", "rb") as file:
        i = 0
        while True:
            rois = np.load(file)
            if i == timestamp:
                return rois[roi_id]

            i += 1

        assert False


def image_buffer_generator():
    bead_coordinates = np.loadtxt(f"{DATA_DIRECTORY}/beadpos_xy.txt", dtype=int)

    with open(f"{DATA_DIRECTORY}/rois.npy", "rb") as file:
        subimages = np.load(file)

        roi_height, roi_width = subimages[0].shape
        roi_coordinates = bead_coordinates - np.array([roi_width // 2, roi_height // 2])
        fill_image_shape = (
            2016,
            2560,
        )

        roi_coordinates[:, [0, 1]] = roi_coordinates[:, [1, 0]]

        images = np.zeros((NUM_IMAGES, *fill_image_shape), dtype=cupy.uint16)
        reconstructed_image = np.zeros(fill_image_shape, dtype=cupy.uint16)

    with open(f"{DATA_DIRECTORY}/rois.npy", "rb") as file:
        try:
            while True:
                for buffer_image_id in range(NUM_IMAGES):
                    sub_images = np.load(file)

                    for bead_image, (row, col) in zip(sub_images, roi_coordinates):
                        reconstructed_image[
                            row : row + roi_height,
                            col : col + roi_width,
                        ] = bead_image

                    images[buffer_image_id, :, :] = reconstructed_image

                yield cupy.array(images)
        except Exception:
            pass


def test_tracker_real_lazy(
    zstacks: cupy.ndarray,
    roi_coordinates: cupy.ndarray,
):
    tracker = Tracker(
        NUM_IMAGES,
        roi_coordinates,
        ROI_SIZE,
        zstacks,
        1,
        ROI_SIZE // 4,
        (ROI_SIZE // 4) * 3,
        100,
        3,
        1,
        ROI_SIZE // 4,
        ROI_SIZE // 4,
        100,
    )

    all_yx_coordinates = []
    all_z_values = []

    traces = []
    with open(f"{DATA_DIRECTORY}/traces.npy", "rb") as file:
        header = cupy.load(file)

        try:
            while True:
                traces.append(cupy.load(file))
        except EOFError:
            pass
    traces = cupy.concatenate(traces)

    for image_buffer in image_buffer_generator():
        tracker.calculate(image_buffer)
        bead_coordinates = tracker.get_calculated_yx()
        z_values = tracker.get_calculated_z()

        all_yx_coordinates.append(bead_coordinates.copy())
        all_z_values.append(z_values.copy())

    all_yx_coordinates = cupy.concatenate(all_yx_coordinates)
    all_z_values = cupy.concatenate(all_z_values)

    plt.ion()

    plot_yx_values(all_yx_coordinates, roi_coordinates, traces)
    plot_z_values(all_z_values, traces)

    plt.ioff()
    animate_y_values(all_yx_coordinates, roi_coordinates, traces)

    input("Press Enter to exit...")


def animate_y_values(
    yx_coordinates: cupy.ndarray, roi_coordinates: cupy.ndarray, traces: cupy.ndarray
):
    yx_in_roi_coordinates = yx_coordinates - roi_coordinates

    num_frames = yx_coordinates.shape[0]

    figure, axes = plt.subplots(2, 4, sharex=True, sharey=True)
    axes = axes.flatten()
    roi_getter = RoiGetter()
    images = [
        axis.imshow(roi_getter.get_roi_np(0, bead_id), cmap="gray", animated=True)
        for axis, bead_id in zip(axes, BEADS_TO_VISUALIEZ)
    ]

    circles_calculated = []
    for axis, bead_id in zip(axes, BEADS_TO_VISUALIEZ):
        [y, x] = yx_in_roi_coordinates[0, bead_id].get()
        circle = matplotlib.patches.Circle(
            (x, y),
            radius=1,
            edgecolor="green",
            facecolor="none",
            lw=2,
        )
        axis.add_patch(circle)
        circles_calculated.append(circle)

    circles_real = []
    for axis, bead_id in zip(axes, BEADS_TO_VISUALIEZ):
        [x, y] = traces[0, bead_id, 0:2]
        circle = matplotlib.patches.Circle(
            (x + ROI_SIZE / 2, y + ROI_SIZE / 2),
            radius=1,
            edgecolor="red",
            facecolor="none",
            lw=2,
        )
        axis.add_patch(circle)
        circles_real.append(circle)

    # Animation update function
    def update(frame_id):
        print(frame_id)
        rois = roi_getter.get_roi_np(frame_id, BEADS_TO_VISUALIEZ)
        for bead_id, roi, image, circle_calculated, circle_real in zip(
            BEADS_TO_VISUALIEZ, rois, images, circles_calculated, circles_real
        ):
            image.set_array(roi)

            [y, x] = yx_in_roi_coordinates[frame_id, bead_id].get()
            circle_calculated.center = (x, y)

            [x, y] = traces[frame_id, bead_id, 0:2]
            circle_real.center = (x + ROI_SIZE / 2, y + ROI_SIZE / 2)

        return images + circles_calculated + circles_real

    animation = matplotlib.animation.FuncAnimation(
        figure, update, frames=range(0, num_frames, 30), blit=True, repeat=True
    )

    plt.show()


def plot_yx_values(
    yx_coordinates: cupy.ndarray, roi_coordinates: cupy.ndarray, traces: cupy.ndarray
):
    yx_in_roi_coordinates = yx_coordinates - roi_coordinates

    figure, axes = plt.subplots(2, 4, sharex=True, sharey=True)
    for axis, bead_id in zip(axes.reshape(-1), BEADS_TO_VISUALIEZ):
        axis.plot(
            yx_in_roi_coordinates[:, bead_id, 0].get() - ROI_SIZE / 2,
            label="Calculated",
            color="g",
        )
        axis.plot(
            traces[:, bead_id, 1].get(),
            label="Expected",
            color="r",
        )

    plt.title("Y COORDINATES")
    plt.show()
    plt.draw()
    plt.legend()
    plt.pause(0.001)

    figure, axes = plt.subplots(2, 4, sharex=True, sharey=True)
    for axis, bead_id in zip(axes.reshape(-1), BEADS_TO_VISUALIEZ):
        axis.plot(
            yx_in_roi_coordinates[:, bead_id, 1].get() - ROI_SIZE / 2,
            label="Calculated",
            color="g",
        )
        axis.plot(
            traces[:, bead_id, 0].get(),
            label="Expected",
            color="r",
        )

    # plt.ylim([-5, 5])
    plt.title("X COORDINATES")
    plt.show()
    plt.draw()
    plt.legend()
    plt.pause(0.001)

    # with open(f"{DATA_DIRECTORY}/all_rois.npy", "rb") as file:
    #     subimages = cupy.load(file)
    #
    # # time_steps_to_show = cupy.array([800, 820, 840, 860, 880, 900, 920, 940, 960, 968])
    # time_steps_to_show = cupy.array(
    #     [
    #         18611,
    #         18612,
    #         18613,
    #         18614,
    #         18615,
    #         18616,
    #         18617,
    #         18618,
    #         18619,
    #         18620,
    #     ]
    # )
    #
    # fig, axes = plt.subplots(2, 5, sharex=True, sharey=True)
    #
    # for axis, yx_in_roi_coordinate, roi_image in zip(
    #     axes.flatten(),
    #     yx_in_roi_coordinates[time_steps_to_show, 11],
    #     subimages[time_steps_to_show, 11],
    # ):
    #     axis.imshow(roi_image.get())
    #
    #     # Create and add the circle
    #     circle = matplotlib.patches.Circle(
    #         (
    #             yx_in_roi_coordinate[1].get(),
    #             yx_in_roi_coordinate[0].get(),
    #         ),
    #         1,
    #         edgecolor="red",
    #         facecolor="none",
    #         linewidth=2,
    #     )
    #     axis.add_patch(circle)
    #
    # # plt.tight_layout()
    # # plt.show()
    # plt.draw()
    # plt.pause(0.001)


def plot_z_values(z_values: cupy.ndarray, traces: cupy.ndarray):
    figure, axes = plt.subplots(2, 4, sharex=True, sharey=True)

    for axis, bead_id in zip(axes.reshape(-1), BEADS_TO_VISUALIEZ):
        axis.plot(z_values[:, bead_id].get() * Z_CORRECTION)
        axis.plot(traces[:, bead_id, 2].get())

        # relative_z = z_values[:, bead_id] - z_values[:, REFERENCE_BEAD_ID]
        # axis.plot(relative_z.get())

    # plt.ylim([3, 5])
    # plt.ylim([70, 85])
    plt.ylim([0, 100])
    plt.show()


def test_tracker_time(
    camera_image: cupy.ndarray,
):
    images = cupy.repeat(cupy.expand_dims(camera_image, axis=0), NUM_IMAGES, axis=0)

    num_rois = 40
    height, width = camera_image.shape
    roi_coordinates = make_roi_coordinates(num_rois, height, width, ROI_SIZE)

    num_z_values = 100
    zstacks = cupy.zeros(
        (num_rois, num_z_values, ROI_SIZE, ROI_SIZE), dtype=cupy.float32
    )

    tracker = Tracker(
        NUM_IMAGES,
        roi_coordinates,
        ROI_SIZE,
        zstacks,
        1,
        ROI_SIZE // 4,
        (ROI_SIZE // 4) * 3,
        100,
        3,
        1,
        ROI_SIZE // 4,
        ROI_SIZE // 4,
        100,
    )

    total_elapsed = 0
    num_iters = 1000
    s2 = cupy.cuda.Stream()

    # Warmup
    for _ in range(10):
        tracker.calculate(images)

    for _ in range(num_iters):
        e1 = cupy.cuda.Event()
        e1.record()
        tracker.calculate(images)
        e2 = cupy.cuda.get_current_stream().record()

        s2.wait_event(e2)

        e2.synchronize()
        t = cupy.cuda.get_elapsed_time(e1, e2)
        total_elapsed += t
        print(f"ELAPSED: {t}ms")

    print(f"AVERAGE ELAPSED: {total_elapsed / (num_iters - 1)}ms")


def test_tracker_measure_buffer_size(
    camera_image: cupy.ndarray,
):
    cupy.random.seed(42)

    buffer_sizes = [
        1,
        10,
        100,
        200,
        300,
        400,
    ]
    num_roises = [
        100,
        200,
        300,
        400,
        500,
        600,
    ]

    min_radius = 1
    max_radius = ROI_SIZE / 4
    number_of_qi_angle_steps = 100
    number_of_qi_radial_steps = ROI_SIZE // 4
    number_of_qi_iterations = 3

    for num_rois in num_roises:
        data = {}
        for num_images in buffer_sizes:
            images = cupy.repeat(
                cupy.expand_dims(camera_image, axis=0), num_images, axis=0
            )

            height, width = camera_image.shape
            roi_coordinates = make_roi_coordinates(num_rois, height, width, ROI_SIZE)

            num_z_values = 100
            zstacks = cupy.zeros(
                (num_rois, num_z_values, ROI_SIZE, ROI_SIZE), dtype=cupy.float32
            )
            tracker = Tracker(
                num_images,
                roi_coordinates,
                ROI_SIZE,
                zstacks,
                min_radius,
                max_radius,
                number_of_qi_radial_steps,
                number_of_qi_angle_steps,
                number_of_qi_iterations,
                1,
                ROI_SIZE // 4,
                ROI_SIZE // 4,
                100,
            )

            total_elapsed = 0
            num_iters = 1000
            s2 = cupy.cuda.Stream()

            # Warmup
            for _ in range(10):
                tracker.calculate(images)

            elapsed_times = []
            for _ in range(num_iters):
                e1 = cupy.cuda.Event()
                e1.record()
                e2 = cupy.cuda.get_current_stream().record()

                s2.wait_event(e2)
                with s2:
                    tracker.calculate(images)

                e2.synchronize()
                t = cupy.cuda.get_elapsed_time(e1, e2)
                total_elapsed += t
                print(f"ELAPSED: {t}ms")
                elapsed_times.append(total_elapsed)

            print(f"AVERAGE ELAPSED: {total_elapsed / (num_iters - 1)}ms")
            data[num_images] = elapsed_times

        data["parameters"] = {
            "num_rois": num_rois,
            "roi_size": ROI_SIZE,
            "max_radial": max_radius,
            "num_radial_steps": number_of_qi_radial_steps,
            "num_angle_steps": number_of_qi_angle_steps,
        }

        file_name = ",".join(
            [f"{key}={value}" for key, value in data["parameters"].items()]
        )

        with open(f"test-results/buffer-sizes/{file_name}.json", "w") as file:
            json.dump(data, file)
