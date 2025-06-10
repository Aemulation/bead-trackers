import cupy

from src.trackers.bead_tracker.center_of_mass import CenterOfMass


def test_calculate_yx_full_image(rings: cupy.ndarray):
    [image_height, image_width] = rings.shape
    assert image_height == image_width
    image_size = image_height

    mock_center_of_mass = CenterOfMass(1, 1, image_size)
    (center_of_masses, averages) = mock_center_of_mass.calculate_yx(
        cupy.expand_dims(rings, axis=0), cupy.array([[0, 0]], dtype=cupy.uint32)
    )
    assert center_of_masses.shape[0] == 1
    assert center_of_masses.shape[1] == 1
    [center_of_mass_y, center_of_mass_x] = center_of_masses[0][0]
    assert cupy.isclose(center_of_mass_y, image_height / 2, atol=1)
    assert cupy.isclose(center_of_mass_x, image_width / 2, atol=1)


def test_calculate_yx_partial_image(rings: cupy.ndarray):
    [image_height, image_width] = rings.shape
    roi_size = 100

    roi_y = (image_height - roi_size) // 2
    roi_x = (image_width - roi_size) // 2

    mock_center_of_mass = CenterOfMass(1, 1, roi_size)
    (center_of_masses, averages) = mock_center_of_mass.calculate_yx(
        cupy.expand_dims(rings, axis=0), cupy.array([[roi_y, roi_x]], dtype=cupy.uint32)
    )
    assert center_of_masses.shape[0] == 1
    assert center_of_masses.shape[1] == 1
    [center_of_mass_y, center_of_mass_x] = center_of_masses[0][0]
    assert cupy.isclose(center_of_mass_y, roi_y + roi_size // 2, atol=1)
    assert cupy.isclose(center_of_mass_x, roi_x + roi_size // 2, atol=1)


def test_calculate_yx_full_image_real(roi_image: cupy.ndarray):
    [image_height, image_width] = roi_image.shape
    assert image_height == image_width
    image_size = image_height

    mock_center_of_mass = CenterOfMass(1, 1, image_size)
    (center_of_masses, averages) = mock_center_of_mass.calculate_yx(
        cupy.expand_dims(roi_image, axis=0), cupy.array([[0, 0]], dtype=cupy.uint32)
    )
    assert center_of_masses.shape[0] == 1
    assert center_of_masses.shape[1] == 1
    [center_of_mass_y, center_of_mass_x] = center_of_masses[0][0]

    expected_yx = [34.1, 23.5]
    assert cupy.isclose(center_of_mass_y, expected_yx[0], atol=0.2 * image_height)
    assert cupy.isclose(center_of_mass_x, expected_yx[1], atol=0.2 * image_width)


def test_calculate_yx_full_image_real_nested(roi_image: cupy.ndarray):
    [image_height, image_width] = roi_image.shape
    assert image_height == image_width
    image_size = image_height

    num_rois = 1
    num_images = 100
    roi_images = cupy.repeat(roi_image, num_images, axis=0).reshape(
        (num_images, image_height, image_width)
    )

    mock_center_of_mass = CenterOfMass(num_images, num_rois, image_size)
    (center_of_masses_nested, averages) = mock_center_of_mass.calculate_yx(
        roi_images, cupy.array([[0, 0]], dtype=cupy.uint32)
    )

    assert center_of_masses_nested.shape == (num_images, num_rois, 2)

    for center_of_masses in center_of_masses_nested:
        assert center_of_masses.shape[0] == num_rois
        [center_of_mass_y, center_of_mass_x] = center_of_masses[0]

        expected_yx = [22.1, 23.5]
        assert cupy.isclose(center_of_mass_y, expected_yx[0], atol=0.2 * image_height)
        assert cupy.isclose(center_of_mass_x, expected_yx[1], atol=0.2 * image_width)
