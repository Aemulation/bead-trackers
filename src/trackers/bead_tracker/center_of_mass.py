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

        self.__range = cupy.arange(self.__roi_size, dtype=cupy.int32)

        self.__offset_y, self.__offset_x = cupy.meshgrid(
            cupy.arange(self.__roi_size), cupy.arange(self.__roi_size), indexing="ij"
        )  # (K, K)

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

    # def __calculate_averages(
    #     self,
    #     images: cupy.ndarray,
    #     roi_coordinates: cupy.ndarray,
    # ) -> cupy.ndarray:
    #     assert roi_coordinates.dtype == cupy.uint32
    #
    #     Y = roi_coordinates[:, 0, None, None] + self.__range
    #     X = roi_coordinates[:, 1, None, None] + self.__range
    #
    #     num_images = images.shape[0]
    #     rois = images[:, Y, X]
    #     num_rois = roi_coordinates.shape[0]
    #     cupy.mean(
    #         rois.reshape((num_images, num_rois, -1)),
    #         axis=2,
    #         out=self.__buffers.averages_buffer,
    #     )
    #     cupy.std(
    #         rois.reshape((num_images, num_rois, -1)),
    #         axis=2,
    #         out=self.__buffers.std_buffer,
    #     )
    #
    #     return (self.__buffers.averages_buffer, self.__buffers.std_buffer)

    # def calculate_yx(
    #     self, images: cupy.ndarray, roi_coordinates: cupy.ndarray
    # ) -> cupy.ndarray:
    #     (averages, stds) = self.__calculate_averages(images, roi_coordinates)
    #     N, H, W = images.shape
    #     R = roi_coordinates.shape[0]
    #     K = self.__roi_size
    #
    #     # Create a grid of offsets for ROI block (KxK)
    #     # offset_y, offset_x = cupy.meshgrid(
    #     #     cupy.arange(K), cupy.arange(K), indexing="ij"
    #     # )  # (K, K)
    #
    #     # Compute absolute ROI positions for each ROI
    #     roi_x = roi_coordinates[:, 0]  # (R,)
    #     roi_y = roi_coordinates[:, 1]  # (R,)
    #     roi_grid_x = roi_x[:, None, None] + self.__offset_x  # (R, K, K)
    #     roi_grid_y = roi_y[:, None, None] + self.__offset_y  # (R, K, K)
    #
    #     # Expand across images: get (N, R, K, K) indices
    #     roi_grid_x = cupy.broadcast_to(roi_grid_x, (N, R, K, K))
    #     roi_grid_y = cupy.broadcast_to(roi_grid_y, (N, R, K, K))
    #
    #     # Gather pixel values using fancy indexing
    #     images_flat = images[:, None]  # (N, 1, H, W)
    #     pixel_values = images_flat[
    #         cupy.arange(N)[:, None, None, None],
    #         cupy.zeros(
    #             (N, R, K, K), dtype=int
    #         ),  # singleton index for dimension alignment
    #         roi_grid_y,
    #         roi_grid_x,
    #     ]  # shape: (N, R, K, K)
    #
    #     # Compute II values
    #     roi_avg = averages[:, :, None, None]  # (N, R, 1, 1)
    #     roi_std = stds[:, :, None, None]  # (N, R, 1, 1)
    #
    #     II = cupy.maximum(
    #         0.0, cupy.abs(pixel_values.astype(cupy.float32) - roi_avg) - roi_std
    #     )  # (N, R, K, K)
    #
    #     # Center of mass
    #     y_coords = self.__offset_y[None, None, :, :]  # (1, 1, K, K)
    #     x_coords = self.__offset_x[None, None, :, :]  # (1, 1, K, K)
    #
    #     total_II = II.sum(axis=(2, 3))  # (N, R)
    #     total_y_II = (II * y_coords).sum(axis=(2, 3))  # (N, R)
    #     total_x_II = (II * x_coords).sum(axis=(2, 3))  # (N, R)
    #
    #     center_y = total_y_II / (total_II) + roi_y[None, :]  # (N, R)
    #     center_x = total_x_II / (total_II) + roi_x[None, :]  # (N, R)
    #
    #     center_of_mass = cupy.stack((center_y, center_x), axis=-1)  # (N, R, 2)
    #     return center_of_mass.astype(cupy.float32), averages

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
