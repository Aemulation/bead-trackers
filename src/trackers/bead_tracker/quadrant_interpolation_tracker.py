import os
import numpy as np
import cupy
from typing import cast

from src.trackers.bead_tracker.cross_correlation import cross_correlate_nested_1d
from src.trackers.bead_tracker.quadratic_polynomial_fitter import (
    QuadraticPolynomialFitter,
)
from src.trackers.bead_tracker.radial_profiler import (
    RadialProfiler,
    RadialProfilerConfig,
)

HERE = os.path.dirname(__file__)


class CudaModules:
    def __init__(self) -> None:
        with open(f"{HERE}/cuda_kernels/polynomial_fitter.cu") as source_file:
            code = source_file.read()

        self.__polynomial_fitter = cupy.RawModule(
            code=code,
            options=("-std=c++11",),
            name_expressions=[
                "copy_relevant_points_max",
            ],
        )
        self.__polynomial_fitter.compile()

    def copy_relevant_points_max(self) -> cupy.RawKernel:
        return self.__polynomial_fitter.get_function("copy_relevant_points_max")


class Buffers:
    points_table: cupy.ndarray
    starting_points_table: cupy.ndarray
    polynomial_coefficients_table: cupy.ndarray
    positions: cupy.ndarray

    def __init__(self, num_images: int, num_beads: int, num_points: int) -> None:
        self.points_table = cupy.empty(
            (num_images, num_beads, num_points), dtype=cupy.float32
        )
        self.starting_points_table = cupy.empty(
            (num_images, num_beads), dtype=cupy.float32
        )
        self.polynomial_coefficients_table = cupy.empty(
            (num_images, num_beads, 3), dtype=cupy.float32
        )
        self.positions = cupy.empty((num_images, num_beads), dtype=cupy.float32)


class QuadrantInterpolationTracker:
    def __init__(
        self,
        num_images: int,
        roi_coordinates: cupy.ndarray,
        roi_size: int,
    ) -> None:
        # TODO: Make configurable.
        radial_profiler_configs = [
            # RadialProfilerConfig(
            #     min_radius=1,
            #     max_radius=25,
            #     num_radial_steps=25 * 3,
            #     num_angle_steps=110,
            #     start_angle_radians=np.pi / 2 * quadrant_id,
            #     end_angle_radians=np.pi / 2 * (quadrant_id + 1),
            #     normalize=False,
            # )
            RadialProfilerConfig(
                min_radius=1,
                max_radius=roi_size / 4,
                num_radial_steps=(roi_size // 4 - 1) * 3,
                num_angle_steps=100,
                start_angle_radians=np.pi / 2 * quadrant_id,
                end_angle_radians=np.pi / 2 * (quadrant_id + 1),
                normalize=False,
            )
            for quadrant_id in range(4)
        ]
        self.__num_images = num_images
        self.__num_beads = roi_coordinates.shape[0]

        self.__radial_profiler_top_right = RadialProfiler(
            radial_profiler_configs[0],
            num_images,
            roi_coordinates,
            roi_size,
        )
        self.__radial_profiler_top_left = RadialProfiler(
            radial_profiler_configs[1],
            num_images,
            roi_coordinates,
            roi_size,
        )
        self.__radial_profiler_bottom_left = RadialProfiler(
            radial_profiler_configs[2],
            num_images,
            roi_coordinates,
            roi_size,
        )
        self.__radial_profiler_bottom_right = RadialProfiler(
            radial_profiler_configs[3],
            num_images,
            roi_coordinates,
            roi_size,
        )

        self.__least_squares_fit_weights = cast(
            cupy.ndarray,
            cupy.array([0.14, 0.5, 0.85, 1.0, 0.85, 0.5, 0.14]),
        )
        self.__num_points = self.__least_squares_fit_weights.shape[0]
        self.__quadratic_polynomial_fitter = QuadraticPolynomialFitter(
            self.__least_squares_fit_weights,
        )

        self.__cuda_modules = CudaModules()

        self.__buffers = Buffers(num_images, self.__num_beads, self.__num_points)

    @staticmethod
    def __fft_cross_correlate(intensity_profile: cupy.ndarray) -> cupy.ndarray:
        return cross_correlate_nested_1d(
            intensity_profile, cupy.flip(intensity_profile, axis=0)
        )

    def __get_relevant_points(self, match_scores_table: cupy.ndarray) -> cupy.ndarray:
        num_beads, num_z_planes = match_scores_table.shape

        num_threads = (16,)
        num_blocks = (num_beads // num_threads[0] + 1,)

        copy_relevant_points_max_kernel = self.__cuda_modules.copy_relevant_points_max()
        copy_relevant_points_max_kernel(
            num_blocks,
            num_threads,
            (
                match_scores_table,
                num_beads,
                num_z_planes,
                self.__num_points,
                self.__buffers.points_table,
                self.__buffers.starting_points_table,
            ),
        )

        return self.__buffers.points_table, self.__buffers.starting_points_table

    def __calculate_delta_position(self, input: cupy.ndarray) -> cupy.ndarray:
        points_table, starting_points = self.__get_relevant_points(input)
        points_table = points_table.reshape(
            self.__num_images * self.__num_beads, self.__num_points
        )
        starting_points = starting_points.reshape(self.__num_images * self.__num_beads)

        coefficients = self.__quadratic_polynomial_fitter.fit_2d(points_table)
        positions = starting_points + cupy.squeeze(
            self.__quadratic_polynomial_fitter.get_top(coefficients), axis=-1
        ).astype(cupy.float32)

        delta_r = (positions - input.shape[1] / 2) * (cupy.pi / 4)
        delta_position = delta_r * self.__radial_profiler_top_right.delta_radial()
        return delta_position

    def calculate_yx(
        self,
        images: cupy.ndarray,
        approximated_bead_coordinates: cupy.ndarray,
        averages: cupy.ndarray,
    ):
        """
        Use quadrant interpolation to calculate the (y, x) position of the beads.

        `approximated_bead_coordinates`: Approximated bead coordinates in (y, x) coordinates.
        """
        assert images.shape[0] == approximated_bead_coordinates.shape[0]
        num_images, num_beads, _ = approximated_bead_coordinates.shape

        radial_profiles_top_right = self.__radial_profiler_top_right.profile(
            images, approximated_bead_coordinates, averages
        )
        radial_profiles_top_left = self.__radial_profiler_top_left.profile(
            images, approximated_bead_coordinates, averages
        )
        radial_profiles_bottom_left = self.__radial_profiler_bottom_left.profile(
            images, approximated_bead_coordinates, averages
        )
        radial_profiles_bottom_right = self.__radial_profiler_bottom_right.profile(
            images, approximated_bead_coordinates, averages
        )

        radial_profiles_right = radial_profiles_top_right + radial_profiles_bottom_right
        radial_profiles_left = radial_profiles_top_left + radial_profiles_bottom_left
        radial_profiles_top = radial_profiles_top_right + radial_profiles_top_left
        radial_profiles_bottom = (
            radial_profiles_bottom_left + radial_profiles_bottom_right
        )

        # # TODO: These copies should not be necessary.
        intensity_profile_x = cupy.concatenate(
            (cupy.flip(radial_profiles_left, axis=2), radial_profiles_right), axis=2
        ).copy()
        assert intensity_profile_x.dtype == cupy.float32
        intensity_profile_y = cupy.concatenate(
            (cupy.flip(radial_profiles_bottom, axis=2), radial_profiles_top),
            axis=2,
        ).copy()
        assert intensity_profile_y.dtype == cupy.float32
        # intensity_profile_x = cupy.concatenate(
        #     (cupy.flip(radial_profiles_left, axis=2), radial_profiles_right), axis=2
        # )
        # assert intensity_profile_x.dtype == cupy.float32
        # intensity_profile_y = cupy.concatenate(
        #     (cupy.flip(radial_profiles_bottom, axis=2), radial_profiles_top),
        #     axis=2,
        # )
        # assert intensity_profile_y.dtype == cupy.float32

        num_images, num_beads, num_profiles = intensity_profile_x.shape
        X = QuadrantInterpolationTracker.__fft_cross_correlate(
            intensity_profile_x.reshape((num_images * num_beads, num_profiles))
        )
        Y = QuadrantInterpolationTracker.__fft_cross_correlate(
            intensity_profile_y.reshape((num_images * num_beads, num_profiles))
        )

        delta_x = self.__calculate_delta_position(X).reshape((num_images, num_beads, 1))
        delta_y = self.__calculate_delta_position(Y).reshape((num_images, num_beads, 1))

        return approximated_bead_coordinates + cupy.concatenate(
            (delta_y, delta_x), axis=2
        )
