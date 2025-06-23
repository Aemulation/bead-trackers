from abc import abstractmethod
from typing import Callable, Protocol
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


class TrackerFactoryProtocol(Protocol):
    @abstractmethod
    def __init__(self, **kwargs) -> None: ...

    @abstractmethod
    def create(self) -> TrackerProtocol: ...


class TrackerFactoryClassRegistry:
    registry: dict[str, type[TrackerFactoryProtocol]] = {}

    @classmethod
    def register(cls, tracker_factory_name: str) -> Callable:
        def inner_wrapper(
            wrapped_factory: type[TrackerFactoryProtocol],
        ) -> type[TrackerFactoryProtocol]:
            cls.registry[tracker_factory_name] = wrapped_factory
            return wrapped_factory

        return inner_wrapper

    @classmethod
    def create(cls, tracker_name: str, **kwargs) -> TrackerFactoryProtocol:
        try:
            tracker_factory = cls.registry[tracker_name]
            return tracker_factory(**kwargs)
        except KeyError:
            raise ValueError(
                f"Tracker {tracker_name} not found, available trackers are: {','.join(cls.registry.keys())}"
            )
