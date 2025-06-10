import cupy
from typing import cast

import pytest

from src.trackers.bead_tracker.radial_profiler import RadialProfilerConfig

NUM_Z_LAYERS = 11
NUM_RADIALS = 5


@pytest.fixture
def mock_roi_image(
    image_size=256, num_circles=10, circle_radius=0.6, dtype=cupy.uint16
):
    x, y = cupy.meshgrid(
        cupy.linspace(-1, 1, image_size), cupy.linspace(-1, 1, image_size)
    )
    r = cupy.sqrt(x**2 + y**2)

    rings = cupy.cos(r * 2 * cupy.pi * num_circles)

    # Blend the rings with the noisy background
    fade_mask = cupy.clip((circle_radius - r) / (circle_radius * 0.3), 0, 1)
    background_noise = cupy.random.normal(0.0, 0.2, size=r.shape)
    rings = rings * fade_mask + (1 - fade_mask) * background_noise

    rings = (rings + 1) * 0.5

    return (rings * cupy.iinfo(dtype).max).astype(dtype)


@pytest.fixture
def roi_image():
    return cast(cupy.ndarray, cupy.load("images/extracted_rois.npy")).astype(
        cupy.uint16
    )[14:64, 28:78]


@pytest.fixture
def camera_image():
    return cast(cupy.ndarray, cupy.load("images/image.npy")).astype(cupy.uint16)


@pytest.fixture
def rings(size=256, num_rings=10, dtype=cupy.uint16):
    x = cupy.linspace(-1, 1, size)
    y = cupy.linspace(-1, 1, size)
    X, Y = cupy.meshgrid(x, y)
    radius = cupy.sqrt(X**2 + Y**2)

    rings = 0.5 * (1 + cupy.sign(cupy.sin(2 * cupy.pi * num_rings * radius)))

    return (rings * cupy.iinfo(dtype).max).astype(dtype)


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
