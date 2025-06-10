import numpy as np
import cupy
import pytest

from src.trackers.bead_tracker.radial_profiler import (
    RadialProfiler,
    RadialProfilerConfig,
)


@pytest.fixture
def radial_profiler_config():
    return RadialProfilerConfig(3.2, 256 / 2 - 3.2 * 3, 10 * 2 - 2, 10, normalize=False)


def test_radial_profile_8_bit(radial_profiler_config: RadialProfilerConfig, rings):
    num_images = 1

    image_height, image_width = rings.shape
    assert image_height == image_width
    image_size = image_height
    centre_y, centre_x = np.array([image_height / 2, image_width / 2], dtype=np.float32)

    roi_coordinates = cupy.array([[0, 0], [0, 0]], dtype=cupy.uint32)
    bead_coordinates = cupy.array([[centre_y, centre_x], [centre_y, centre_x]])
    bead_coordinates = cupy.repeat(bead_coordinates, num_images, axis=0).reshape(
        (
            num_images,
            2,
            2,
        )
    )

    radial_profiler = RadialProfiler(
        radial_profiler_config,
        num_images,
        roi_coordinates,
        image_size,
    )

    rings = rings.astype(cupy.uint8)
    rings = cupy.repeat(rings, num_images, axis=0).reshape(
        (num_images, image_height, image_width)
    )

    averages = cupy.zeros((num_images, 2), dtype=cupy.float32)

    radial_profiles_nested = radial_profiler.profile(rings, bead_coordinates, averages)
    assert radial_profiles_nested.shape[0] == num_images
    assert radial_profiles_nested.shape[1] == bead_coordinates.shape[1]

    for i, profile in enumerate(radial_profiles_nested[0][0]):
        if i % 2 == 0:
            assert profile == np.iinfo(np.uint8).max
        if i % 2 == 1:
            assert profile == np.iinfo(np.uint8).min


def test_radial_profile_16_bit(radial_profiler_config: RadialProfilerConfig, rings):
    num_images = 1

    image_height, image_width = rings.shape
    assert image_height == image_width
    image_size = image_height
    centre_y, centre_x = np.array([image_height / 2, image_width / 2], dtype=np.float32)

    roi_coordinates = cupy.array([[0, 0], [0, 0]], dtype=cupy.uint32)
    bead_coordinates = cupy.array([[centre_y, centre_x], [centre_y, centre_x]])
    bead_coordinates = cupy.repeat(bead_coordinates, num_images, axis=0).reshape(
        (
            num_images,
            2,
            2,
        )
    )

    radial_profiler = RadialProfiler(
        radial_profiler_config, num_images, roi_coordinates, image_size
    )

    rings = rings.astype(cupy.uint16)
    rings = cupy.repeat(rings, num_images, axis=0).reshape(
        (num_images, image_height, image_width)
    )

    averages = cupy.zeros((num_images, 2), dtype=cupy.float32)

    radial_profiles_nested = radial_profiler.profile(rings, bead_coordinates, averages)
    assert radial_profiles_nested.shape[0] == num_images
    assert radial_profiles_nested.shape[1] == bead_coordinates.shape[1]

    for i, profile in enumerate(radial_profiles_nested[0][0]):
        if i % 2 == 0:
            assert profile == np.iinfo(np.uint16).max
        if i % 2 == 1:
            assert profile == np.iinfo(np.uint16).min


def test_radial_profile_nested_images(
    radial_profiler_config: RadialProfilerConfig, rings: cupy.ndarray
):
    num_images = 100

    image_height, image_width = rings.shape
    assert image_height == image_width
    image_size = image_height
    centre_y, centre_x = np.array([image_height / 2, image_width / 2], dtype=np.float32)

    roi_coordinates = cupy.array([[0, 0], [0, 0]], dtype=cupy.uint32)
    bead_coordinates = cupy.array([[centre_y, centre_x], [centre_y, centre_x]])
    bead_coordinates = cupy.repeat(
        cupy.expand_dims(bead_coordinates, axis=0), num_images, axis=0
    ).reshape(
        (
            num_images,
            2,
            2,
        )
    )

    radial_profiler = RadialProfiler(
        radial_profiler_config,
        num_images,
        roi_coordinates,
        image_size,
    )

    rings = rings.astype(cupy.uint16)
    rings = cupy.repeat(cupy.expand_dims(rings, axis=0), num_images, axis=0).reshape(
        (num_images, image_height, image_width)
    )
    averages = cupy.zeros((num_images, 2), dtype=cupy.float32)

    for ring in rings:
        assert cupy.all(ring == rings[0])

    radial_profiles_nested = radial_profiler.profile(rings, bead_coordinates, averages)
    assert radial_profiles_nested.shape[0] == num_images
    assert radial_profiles_nested.shape[1] == bead_coordinates.shape[1]

    for radial_profiles in radial_profiles_nested:
        for bead_profile in radial_profiles:
            for i, profile in enumerate(bead_profile):
                if i % 2 == 0:
                    assert profile == np.iinfo(np.uint16).max
                if i % 2 == 1:
                    assert profile == np.iinfo(np.uint16).min


def test_radial_profile_normalize(radial_profiler_config: RadialProfilerConfig, rings):
    num_images = 1

    image_height, image_width = rings.shape
    assert image_height == image_width
    image_size = image_height
    centre_y, centre_x = np.array([image_height / 2, image_width / 2], dtype=np.float32)

    roi_coordinates = cupy.array([[0, 0], [0, 0]], dtype=cupy.uint32)
    bead_coordinates = cupy.array([[centre_y, centre_x], [centre_y, centre_x]])
    bead_coordinates = cupy.repeat(bead_coordinates, num_images, axis=0).reshape(
        (
            num_images,
            2,
            2,
        )
    )

    radial_profiler_config.normalize = True
    radial_profiler = RadialProfiler(
        radial_profiler_config,
        num_images,
        roi_coordinates,
        image_size,
    )

    rings = rings.astype(cupy.uint16)
    rings = cupy.repeat(rings, num_images, axis=0).reshape(
        (num_images, image_height, image_width)
    )
    averages = cupy.ones((num_images, 2), dtype=cupy.float32)

    radial_profiles_nested = radial_profiler.profile(rings, bead_coordinates, averages)
    assert radial_profiles_nested.shape[0] == num_images
    assert radial_profiles_nested.shape[1] == bead_coordinates.shape[1]

    for i, profile in enumerate(radial_profiles_nested[0][0]):
        if i % 2 == 0:
            assert profile == 1
        if i % 2 == 1:
            assert profile == -1
