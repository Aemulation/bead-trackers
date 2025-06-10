import cupy
import os

HERE = os.path.dirname(__file__)


class CudaModule:
    def __init__(self) -> None:
        with open(f"{HERE}/cuda_kernels/center_of_mass.cu") as source_file:
            code = source_file.read()

        self.__raw_module = cupy.RawModule(
            code=code,
            options=("-std=c++11",),
            name_expressions=[
                "center_of_mass<uint8_t>",
                "center_of_mass<uint16_t>",
                "calculate_averages<uint8_t>",
                "calculate_averages<uint16_t>",
            ],
        )
        self.__raw_module.compile()

    def calculate_averages_8bit(self) -> cupy.RawKernel:
        return self.__raw_module.get_function("calculate_averages<uint8_t>")

    def calculate_averages_16bit(self) -> cupy.RawKernel:
        return self.__raw_module.get_function("calculate_averages<uint16_t>")

    def center_of_mass_8bit(self) -> cupy.RawKernel:
        return self.__raw_module.get_function("center_of_mass<uint8_t>")

    def center_of_mass_16bit(self) -> cupy.RawKernel:
        return self.__raw_module.get_function("center_of_mass<uint16_t>")


class Buffers:
    averages_buffer: cupy.ndarray
    std_buffer: cupy.ndarray
    center_of_mass_buffer: cupy.ndarray

    def __init__(self, num_images: int, num_rois: int) -> None:
        self.averages_buffer = cupy.empty((num_images, num_rois), dtype=cupy.float32)
        self.std_buffer = cupy.empty((num_images, num_rois), dtype=cupy.float32)
        self.center_of_mass_buffer = cupy.empty(
            (num_images, num_rois, 2), dtype=cupy.float32
        )


class CenterOfMass:
    def __init__(self, num_images: int, num_rois: int, roi_size: int) -> None:
        self.__roi_size = roi_size

        self.__cuda_modules = CudaModule()

        self.__buffers = Buffers(num_images, num_rois)

    def __calculate_averages(
        self,
        images: cupy.ndarray,
        roi_coordinates: cupy.ndarray,
    ) -> cupy.ndarray:
        assert roi_coordinates.dtype == cupy.uint32
        if images.dtype == cupy.uint8:
            calculate_averages_kernel = self.__cuda_modules.calculate_averages_8bit()
        elif images.dtype == cupy.uint16:
            calculate_averages_kernel = self.__cuda_modules.calculate_averages_16bit()
        else:
            raise ValueError(f"Unsupported pixel datatype: {images.dtype}")

        num_images, image_height, image_width = images.shape
        num_rois = roi_coordinates.shape[0]

        num_threads = (16,)
        num_blocks = (num_rois // num_threads[0] + 1, num_images)

        calculate_averages_kernel(
            num_blocks,
            num_threads,
            (
                images,
                num_images,
                image_height,
                image_width,
                roi_coordinates,
                num_rois,
                self.__roi_size,
                self.__buffers.averages_buffer,
                self.__buffers.std_buffer,
            ),
        )

        return (self.__buffers.averages_buffer, self.__buffers.std_buffer)

    def calculate_yx(
        self, images: cupy.ndarray, roi_coordinates: cupy.ndarray
    ) -> cupy.ndarray:
        (averages, stds) = self.__calculate_averages(images, roi_coordinates)

        if images.dtype == cupy.uint8:
            center_of_mass_kernel = self.__cuda_modules.center_of_mass_8bit()
        elif images.dtype == cupy.uint16:
            center_of_mass_kernel = self.__cuda_modules.center_of_mass_16bit()
        else:
            raise ValueError(f"Unsupported pixel datatype: {images.dtype}")

        num_images, image_height, image_width = images.shape
        num_rois = roi_coordinates.shape[0]

        num_threads = (16,)
        num_blocks = (num_rois // num_threads[0] + 1, num_images)

        center_of_mass_kernel(
            num_blocks,
            num_threads,
            (
                images,
                num_images,
                image_height,
                image_width,
                roi_coordinates,
                num_rois,
                self.__roi_size,
                averages,
                stds,
                self.__buffers.center_of_mass_buffer,
            ),
        )

        return (self.__buffers.center_of_mass_buffer, averages)
