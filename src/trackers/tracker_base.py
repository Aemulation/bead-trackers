from typing import Protocol
import cupy


class TrackerProtocol(Protocol):
    def __init__(
        self,
        num_images_per_buffer: int,
        roi_coordinates: cupy.ndarray,
        roi_size: int,
        lookup_table_images: cupy.ndarray,
        *args,
        **kwargs,
    ) -> None: ...

    def calculate(self, images: cupy.ndarray): ...

    def get_calculated_yx(self) -> cupy.ndarray: ...

    def get_calculated_z(self) -> cupy.ndarray: ...
