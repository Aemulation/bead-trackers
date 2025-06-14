import cupy
import matplotlib.patches
import matplotlib.pyplot as plt
import time

from src.trackers.tracker_base import TrackerProtocol

from src.trackers.bead_tracker.z_lookup_tables import ZLookupTables
from src.trackers.bead_tracker.radial_profiler import (
    RadialProfiler,
    RadialProfilerConfig,
)
from src.trackers.bead_tracker.center_of_mass import CenterOfMass
from src.trackers.bead_tracker.quadrant_interpolation_tracker import (
    QuadrantInterpolationTracker,
)


class Tracker(TrackerProtocol):
    def __init__(
        self,
        num_images_per_buffer: int,
        roi_coordinates: cupy.ndarray,
        roi_size: int,
        lookup_table_images: cupy.ndarray,
        min_qi_radius: float,
        max_qi_radius: float,
        number_of_qi_radial_steps: int,
        number_of_qi_angle_steps: int,
        number_of_qi_iterations: int,
        min_lut_radius: float,
        max_lut_radius: float,
        number_of_lut_radial_steps: int,
        number_of_lut_angle_steps: int,
        *args,
        **kwargs,
    ) -> None:
        assert lookup_table_images.shape[0] == roi_coordinates.shape[0]
        num_rois = roi_coordinates.shape[0]

        self.__bead_coordinates = None
        self.__z_values = None
        self.__roi_coordinates = roi_coordinates
        self.__roi_size = roi_size
        self.__number_of_qi_iterations = number_of_qi_iterations

        self.__center_of_mass = CenterOfMass(
            num_images_per_buffer,
            num_rois,
            roi_size,
        )

        self.__quadrant_interpolation_tracker = QuadrantInterpolationTracker(
            num_images_per_buffer,
            roi_coordinates,
            roi_size,
            min_qi_radius,
            max_qi_radius,
            number_of_qi_radial_steps,
            number_of_qi_angle_steps,
        )

        radial_profiler_config = RadialProfilerConfig(
            min_lut_radius,
            max_lut_radius,
            number_of_lut_radial_steps,
            number_of_lut_angle_steps,
        )

        self.__radial_profiler = RadialProfiler(
            radial_profiler_config, num_images_per_buffer, roi_coordinates, roi_size
        )

        lookup_table_profiles = self.create_lookup_table_profiles(
            lookup_table_images,
            radial_profiler_config,
            min_qi_radius,
            max_qi_radius,
            number_of_qi_radial_steps,
            number_of_qi_angle_steps,
            number_of_qi_iterations,
        )
        num_layers = lookup_table_profiles.shape[1]
        z_values = cupy.linspace(
            0,
            num_layers,
            num_layers,
            dtype=cupy.float32,
        )

        self.__z_lookup_tables = ZLookupTables(
            z_lookup_tables=lookup_table_profiles,
            num_images=num_images_per_buffer,
            z_values=z_values,
            least_squares_fit_weights=cupy.array(
                [0.15, 0.5, 0.85, 1.0, 0.85, 0.5, 0.15]
            ),
        )

    def create_lookup_table_profiles(
        self,
        lookup_table_images: cupy.ndarray,
        radial_profiler_config: RadialProfilerConfig,
        min_qi_radius: float,
        max_qi_radius: float,
        number_of_qi_radial_steps: int,
        number_of_qi_angle_steps: int,
        number_of_qi_iterations: int,
    ) -> cupy.ndarray:
        z_lookup_tables = []
        for image in lookup_table_images:
            images = image.astype(cupy.uint16)
            num_images, image_height, image_width = images.shape
            assert image_height == image_width
            image_size = image_height

            roi_coordinates = cupy.array([[0, 0]], dtype=cupy.uint32)
            num_beads = roi_coordinates.shape[0]

            center_of_mass = CenterOfMass(num_images, num_beads, image_height)
            qi_tracker = QuadrantInterpolationTracker(
                num_images,
                roi_coordinates,
                image_height,
                min_qi_radius,
                max_qi_radius,
                number_of_qi_radial_steps,
                number_of_qi_angle_steps,
            )

            (bead_coordinates, averages) = center_of_mass.calculate_yx(
                images, roi_coordinates
            )
            for _ in range(number_of_qi_iterations):
                bead_coordinates = qi_tracker.calculate_yx(
                    images, bead_coordinates, averages
                )

            radial_profiler = RadialProfiler(
                radial_profiler_config,
                num_images,
                roi_coordinates,
                image_size,
            )
            radial_profiles = radial_profiler.profile(
                images, bead_coordinates, averages
            )

            z_lookup_table = radial_profiles.get().reshape(
                (num_images, radial_profiler_config.num_radial_steps)
            )
            z_lookup_tables.append(z_lookup_table)

        z_lookup_tables = cupy.array(z_lookup_tables)

        return z_lookup_tables.astype(cupy.float32)

    def calculate(self, images: cupy.ndarray):
        (bead_coordinates, averages) = self.__center_of_mass.calculate_yx(
            images, self.__roi_coordinates
        )

        for _ in range(self.__number_of_qi_iterations):
            bead_coordinates = self.__quadrant_interpolation_tracker.calculate_yx(
                images, bead_coordinates, averages
            )

        radial_profiles = self.__radial_profiler.profile(
            images, bead_coordinates, averages
        )

        z_values = self.__z_lookup_tables.compute_z(radial_profiles)

        self.__bead_coordinates = bead_coordinates
        self.__z_values = z_values

    def get_calculated_yx(self) -> cupy.ndarray:
        return self.__bead_coordinates

    def get_calculated_z(self) -> cupy.ndarray:
        return self.__z_values
