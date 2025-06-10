import os
import numpy as np
import cupy
from dataclasses import dataclass

HERE = os.path.dirname(__file__)


@dataclass
class RadialProfilerConfig:
    min_radius: float
    max_radius: float
    num_radial_steps: int

    num_angle_steps: int
    start_angle_radians: float = 0
    end_angle_radians: float = 2 * np.pi

    normalize: bool = True


class CudaModule:
    def __init__(self) -> None:
        with open(f"{HERE}/cuda_kernels/radial_profile.cu") as source_file:
            code = source_file.read()

        self.__raw_module = cupy.RawModule(
            code=code,
            options=("-std=c++11",),
            name_expressions=["radial_profile<uint8_t>", "radial_profile<uint16_t>"],
        )
        self.__raw_module.compile()

    def radial_profile_8bit(self) -> cupy.RawKernel:
        return self.__raw_module.get_function("radial_profile<uint8_t>")

    def radial_profile_16bit(self) -> cupy.RawKernel:
        return self.__raw_module.get_function("radial_profile<uint16_t>")


class Buffers:
    radial_profile_buffer: cupy.ndarray

    normalize_buffer: cupy.ndarray
    normalize_mean_buffer: cupy.ndarray
    normalize_rms_buffer: cupy.ndarray

    def __init__(self, num_images: int, num_beads: int, num_radials: int) -> None:
        self.radial_profile_buffer = cupy.empty(
            (num_images, num_beads, num_radials), dtype=cupy.float32
        )

        self.normalize_buffer = cupy.empty(
            (num_images, num_beads, num_radials), dtype=cupy.float32
        )
        self.normalize_mean_buffer = cupy.empty(
            (num_images, num_beads), dtype=cupy.float32
        )
        self.normalize_rms_buffer = cupy.empty(
            (num_images, num_beads), dtype=cupy.float32
        )


class RadialProfiler:
    def __init__(
        self,
        config: RadialProfilerConfig,
        num_images: int,
        roi_coordinates: cupy.ndarray,
        roi_size: int,
    ) -> None:
        self.__config = config
        self.__roi_coordinates = roi_coordinates
        self.__roi_size = roi_size

        num_beads = roi_coordinates.shape[0]

        self.__radial_position_lookup_table = (
            self.__create_radial_position_lookup_table()
        )
        self.__radial_lookup_table = self.__create_radial_lookup_table()

        self.__cuda_modules = CudaModule()

        self.__buffers = Buffers(num_images, num_beads, config.num_radial_steps)

    def delta_radial(self) -> cupy.float32:
        return self.__radial_lookup_table[1] - self.__radial_lookup_table[0]

    def __radial_profile(
        self, images: cupy.ndarray, bead_positions: cupy.ndarray, averages: cupy.ndarray
    ):
        if images.dtype == cupy.uint8:
            radial_profile_kernel = self.__cuda_modules.radial_profile_8bit()
        elif images.dtype == cupy.uint16:
            radial_profile_kernel = self.__cuda_modules.radial_profile_16bit()
        else:
            raise ValueError(f"Unsupported pixel datatype: {images.dtype}")

        num_images, image_height, image_width = images.shape
        assert bead_positions.shape[0] == num_images
        num_beads = bead_positions.shape[1]
        num_radials = self.__radial_lookup_table.shape[0]
        num_radial_positions = self.__radial_position_lookup_table.shape[0]

        num_threads = (16, 16)
        num_blocks = (
            num_beads // num_threads[0] + 1,
            num_radials // num_threads[1] + 1,
            num_images,
        )

        radial_profile_kernel(
            num_blocks,
            num_threads,
            (
                images,
                num_images,
                image_height,
                image_width,
                self.__roi_coordinates,
                self.__roi_size,
                bead_positions,
                num_beads,
                self.__radial_lookup_table,
                num_radials,
                self.__radial_position_lookup_table,
                num_radial_positions,
                averages,
                self.__buffers.radial_profile_buffer,
            ),
        )

        return self.__buffers.radial_profile_buffer

    def __normalize_radial_profile(self, radial_profiles: cupy.ndarray):
        num_radials = radial_profiles.shape[2]

        mean = cupy.mean(
            radial_profiles, axis=2, out=self.__buffers.normalize_mean_buffer
        )
        meaned_radial_profile = cupy.subtract(
            radial_profiles, mean[..., None], out=self.__buffers.normalize_buffer
        )

        rms_squared_sum = cupy.sum(
            cupy.square(meaned_radial_profile),
            axis=2,
            out=self.__buffers.normalize_rms_buffer,
        )
        total_rms = cupy.sqrt(
            rms_squared_sum / num_radials, out=self.__buffers.normalize_rms_buffer
        )
        normalized_radial_profiles = cupy.divide(
            meaned_radial_profile,
            total_rms[..., None],
            out=self.__buffers.normalize_buffer,
        )

        return normalized_radial_profiles

    def __create_radial_position_lookup_table(self) -> cupy.ndarray:
        angles = cupy.linspace(
            self.__config.start_angle_radians,
            self.__config.end_angle_radians,
            self.__config.num_angle_steps,
            dtype=cupy.float32,
            endpoint=False,
        )

        sin_angles = cupy.sin(angles)
        cos_angles = cupy.cos(angles)

        return cupy.stack((sin_angles, cos_angles), axis=1, dtype=cupy.float32)

    def __create_radial_lookup_table(self) -> cupy.ndarray:
        config = self.__config

        return cupy.linspace(
            config.min_radius,
            config.max_radius,
            config.num_radial_steps,
            dtype=cupy.float32,
            endpoint=False,
        )

    def profile(
        self,
        images: cupy.ndarray,
        bead_coordinates: cupy.ndarray,
        averages: cupy.ndarray,
    ) -> cupy.ndarray:
        assert bead_coordinates.dtype == cupy.float32

        radial_profiles = self.__radial_profile(images, bead_coordinates, averages)
        if self.__config.normalize:
            radial_profiles = self.__normalize_radial_profile(radial_profiles)

        return radial_profiles
