import os
import cupy

from src.trackers.bead_tracker.quadratic_polynomial_fitter import (
    QuadraticPolynomialFitter,
)

HERE = os.path.dirname(__file__)


class CudaModules:
    def __init__(self) -> None:
        with open(f"{HERE}/cuda_kernels/z_lookup_tables.cu") as source_file:
            code = source_file.read()

        self.__z_lookup_tables = cupy.RawModule(
            code=code,
            options=("-std=c++11",),
            name_expressions=[
                "compute_profile_match_scores",
            ],
        )
        self.__z_lookup_tables.compile()

        with open(f"{HERE}/cuda_kernels/polynomial_fitter.cu") as source_file:
            code = source_file.read()

        self.__polynomial_fitter = cupy.RawModule(
            code=code,
            options=("-std=c++11",),
            name_expressions=[
                "copy_relevant_points_min",
            ],
        )
        self.__polynomial_fitter.compile()

    def compute_profile_match_scores(self) -> cupy.RawKernel:
        return self.__z_lookup_tables.get_function("compute_profile_match_scores")

    def copy_relevant_points_min(self) -> cupy.RawKernel:
        return self.__polynomial_fitter.get_function("copy_relevant_points_min")


class Buffers:
    points_table: cupy.ndarray
    starting_points_table: cupy.ndarray
    match_scores_table: cupy.ndarray
    polynomial_coefficients_table: cupy.ndarray
    positions: cupy.ndarray

    def __init__(
        self, num_images: int, num_beads: int, num_points: int, num_z_planes: int
    ) -> None:
        self.points_table = cupy.empty(
            (num_images, num_beads, num_points), dtype=cupy.float32
        )
        self.starting_points_table = cupy.empty(
            (num_images, num_beads), dtype=cupy.float32
        )
        self.match_scores_table = cupy.empty(
            (num_images * num_beads, num_z_planes), dtype=cupy.float32
        )
        self.polynomial_coefficients_table = cupy.empty(
            (num_images * num_beads, 3), dtype=cupy.float32
        )
        self.positions = cupy.empty((num_images, num_beads), dtype=cupy.float32)


class ZLookupTables:
    def __init__(
        self,
        z_lookup_tables: cupy.ndarray,
        z_values: cupy.ndarray,
        num_images: int,
        least_squares_fit_weights=cupy.array([0.5, 0.85, 1.0, 0.85, 0.5]),
    ) -> None:
        self.__num_images = num_images
        self.__z_lookup_tables = z_lookup_tables
        self.__z_values = z_values
        self.__num_beads = z_lookup_tables.shape[0]

        self.__num_points = least_squares_fit_weights.shape[0]
        self.__quadratic_polynomial_fitter = QuadraticPolynomialFitter(
            least_squares_fit_weights
        )
        self.__num_z_values = z_values.shape[0]

        self.__z_interpolation_x = cupy.arange(self.__num_z_values, dtype=cupy.float32)

        self.__cuda_modules = CudaModules()

        self.__buffers = Buffers(
            num_images,
            self.__num_beads,
            least_squares_fit_weights.shape[0],
            self.__num_z_values,
        )

    def __compute_profile_match_scores(
        self, radial_profiles: cupy.ndarray
    ) -> cupy.ndarray:
        num_images, num_beads, num_radials = radial_profiles.shape
        num_z_planes = self.__z_lookup_tables.shape[1]
        assert self.__z_lookup_tables.shape[2] == num_radials
        assert self.__z_lookup_tables.dtype == cupy.float32
        assert radial_profiles.dtype == cupy.float32

        num_threads = (16, 16)
        num_blocks = (
            num_beads // num_threads[0] + 1,
            num_z_planes // num_threads[1] + 1,
            num_images,
        )

        compute_profile_match_scores_kernel = (
            self.__cuda_modules.compute_profile_match_scores()
        )
        compute_profile_match_scores_kernel(
            num_blocks,
            num_threads,
            (
                radial_profiles,
                self.__z_lookup_tables,
                num_beads,
                num_z_planes,
                num_radials,
                self.__buffers.match_scores_table,
            ),
        )

        return self.__buffers.match_scores_table

    def __get_relevant_points(self, match_scores_table: cupy.ndarray) -> cupy.ndarray:
        num_beads, num_z_planes = match_scores_table.shape

        num_threads = (16,)
        num_blocks = (num_beads // num_threads[0] + 1,)

        copy_relevant_points_min_kernel = self.__cuda_modules.copy_relevant_points_min()
        copy_relevant_points_min_kernel(
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

    def compute_z(self, radial_profiles: cupy.ndarray):
        assert radial_profiles.dtype == cupy.float32
        assert radial_profiles.shape[0] == self.__num_images
        assert radial_profiles.shape[1] == self.__num_beads

        assert self.__num_images == radial_profiles.shape[0]
        assert self.__num_beads == self.__z_lookup_tables.shape[0]

        match_scores_table = self.__compute_profile_match_scores(radial_profiles)
        points_table, starting_points = self.__get_relevant_points(
            match_scores_table.reshape(
                (self.__num_images * self.__num_beads, self.__num_z_values)
            )
        )
        points_table = points_table.reshape(
            self.__num_images * self.__num_beads, self.__num_points
        )
        starting_points = starting_points.reshape(self.__num_images * self.__num_beads)

        coefficients = self.__quadratic_polynomial_fitter.fit_2d(points_table)
        z_offsets = self.__quadratic_polynomial_fitter.get_top(coefficients)
        z_offsets = cupy.squeeze(z_offsets, axis=-1)

        computed_z_flattened = cupy.interp(
            starting_points + z_offsets,
            self.__z_interpolation_x,
            self.__z_values,
        )

        computed_z = computed_z_flattened.reshape((self.__num_images, self.__num_beads))
        return computed_z.astype(cupy.float32)
