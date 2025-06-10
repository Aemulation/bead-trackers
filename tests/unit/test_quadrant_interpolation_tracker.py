import cupy
import numpy as np

from src.trackers.bead_tracker.quadrant_interpolation_tracker import (
    QuadrantInterpolationTracker,
)


def test_quadrant_interpolation_tracker(rings: cupy.ndarray):
    num_images = 1
    [image_height, image_width] = rings.shape
    assert image_height == image_width
    images = cupy.repeat(cupy.expand_dims(rings, axis=0), num_images, axis=0)

    roi_coordinates = cupy.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    bead_coordinates_nested = cupy.repeat(
        cupy.array(
            [
                [
                    [image_height // 2 + 0.5, image_width // 2 + 0.5],
                    [image_height // 2 + 0.5, image_width // 2 - 0.5],
                    [image_height // 2 - 0.5, image_width // 2 + 0.5],
                    [image_height // 2 - 0.5, image_width // 2 - 0.5],
                ]
            ],
            dtype=cupy.float32,
        ),
        num_images,
        axis=0,
    )

    quadrant_interpolation_tracker = QuadrantInterpolationTracker(
        num_images, roi_coordinates, image_height
    )

    averages = cupy.expand_dims(cupy.average(rings), axis=0)
    for _ in range(3):
        bead_coordinates_nested = quadrant_interpolation_tracker.calculate_yx(
            images, bead_coordinates_nested, averages
        )

    absolute_tollerance = 0.4
    for bead_coordinates in bead_coordinates_nested:
        for y_coordinate, x_coordinate in bead_coordinates:
            print(f"{y_coordinate} =~ {image_height / 2}")
            assert np.isclose(y_coordinate, image_height / 2, atol=absolute_tollerance)
            print(f"{x_coordinate} =~ {image_width / 2}")
            assert np.isclose(x_coordinate, image_width / 2, atol=absolute_tollerance)
            print()


def test_quadrant_interpolation_tracker_nested(rings: cupy.ndarray):
    num_images = 100
    [image_height, image_width] = rings.shape
    assert image_height == image_width
    images = cupy.repeat(cupy.expand_dims(rings, axis=0), num_images, axis=0)

    roi_coordinates = cupy.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    bead_coordinates_nested = cupy.repeat(
        cupy.array(
            [
                [
                    [image_height // 2 + 0.5, image_width // 2 + 0.5],
                    [image_height // 2 + 0.5, image_width // 2 - 0.5],
                    [image_height // 2 - 0.5, image_width // 2 + 0.5],
                    [image_height // 2 - 0.5, image_width // 2 - 0.5],
                ]
            ],
            dtype=cupy.float32,
        ),
        num_images,
        axis=0,
    )

    quadrant_interpolation_tracker = QuadrantInterpolationTracker(
        num_images, roi_coordinates, image_height
    )

    averages = cupy.expand_dims(cupy.average(rings), axis=0)
    for _ in range(3):
        bead_coordinates_nested = quadrant_interpolation_tracker.calculate_yx(
            images, bead_coordinates_nested, averages
        )

    absolute_tollerance = 0.4
    for bead_coordinates in bead_coordinates_nested:
        for y_coordinate, x_coordinate in bead_coordinates:
            print(f"{y_coordinate} =~ {image_height / 2}")
            assert np.isclose(y_coordinate, image_height / 2, atol=absolute_tollerance)
            print(f"{x_coordinate} =~ {image_width / 2}")
            assert np.isclose(x_coordinate, image_width / 2, atol=absolute_tollerance)
            print()


def test_quadrant_interpolation_tracker_real(
    roi_image: cupy.ndarray,
):
    num_images = 1
    image_height, image_width = roi_image.shape
    assert image_height == image_width
    images = cupy.repeat(cupy.expand_dims(roi_image, axis=0), num_images, axis=0)

    expected_bead_coordinate = cupy.array(
        [22.8, 24.2],
        dtype=cupy.float32,
    )

    roi_coordinates = cupy.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    bead_coordinates_nested = cupy.repeat(
        cupy.array(
            [
                [
                    [22.8, 24.2],
                    [22.8 - 0.2, 24.2],
                    [22.8 + 0.2, 24.2],
                    [22.8, 24.2 - 0.2],
                    [22.8, 24.2 + 0.2],
                ]
            ],
            dtype=cupy.float32,
        ),
        num_images,
        axis=0,
    )
    quadrant_interpolation_tracker = QuadrantInterpolationTracker(
        num_images, roi_coordinates, image_height
    )

    averages = cupy.expand_dims(cupy.average(roi_image), axis=0)
    for _ in range(3):
        bead_coordinates_nested = quadrant_interpolation_tracker.calculate_yx(
            images, bead_coordinates_nested, averages
        )

    absolute_tollerance = 0.3
    for bead_coordinates in bead_coordinates_nested:
        for y_coordinate, x_coordinate in bead_coordinates:
            print(f"{y_coordinate} =~ {expected_bead_coordinate[0]}")
            assert np.isclose(
                y_coordinate, expected_bead_coordinate[0], atol=absolute_tollerance
            )
            print(f"{x_coordinate} =~ {expected_bead_coordinate[1]}")
            assert np.isclose(
                x_coordinate, expected_bead_coordinate[1], atol=absolute_tollerance
            )
            print()
