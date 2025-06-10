import cupy

from src.trackers.bead_tracker.z_lookup_tables import ZLookupTables

from tests.unit.conftest import NUM_Z_LAYERS


def test_single_z_lookup_table(mock_z_lookup_table: cupy.ndarray, mock_z_values):
    z_lookup_tables = ZLookupTables(
        cupy.array([mock_z_lookup_table]),
        mock_z_values,
        1,
        least_squares_fit_weights=cupy.array([0.85, 1.0, 0.85]),
    )

    # Skip z lookup table edges.
    for mock_z_id in range(1, NUM_Z_LAYERS - 1):
        mock_radial_profiles = cupy.array(
            [[mock_z_lookup_table[mock_z_id]]], dtype=cupy.float32
        )
        computed_z_nested = z_lookup_tables.compute_z(mock_radial_profiles)

        assert computed_z_nested.shape[0] == 1
        assert computed_z_nested.dtype == cupy.float32
        assert computed_z_nested[0] == mock_z_values[mock_z_id]


def test_single_z_lookup_table_edges(mock_z_lookup_table: cupy.ndarray, mock_z_values):
    z_lookup_tables = ZLookupTables(
        cupy.array([mock_z_lookup_table]),
        mock_z_values,
        1,
        least_squares_fit_weights=cupy.array([0.85, 1.0, 0.85]),
    )

    mock_radial_profiles = cupy.array([[mock_z_lookup_table[0]]], dtype=cupy.float32)
    computed_z = z_lookup_tables.compute_z(mock_radial_profiles)
    assert computed_z == mock_z_values[0]

    mock_radial_profiles = cupy.array([[mock_z_lookup_table[-1]]], dtype=cupy.float32)
    computed_z = z_lookup_tables.compute_z(mock_radial_profiles)
    assert computed_z == mock_z_values[-1]


def test_multiple_z_lookup_table(mock_z_lookup_table: cupy.ndarray, mock_z_values):
    z_lookup_tables = ZLookupTables(
        cupy.array([mock_z_lookup_table] * NUM_Z_LAYERS),
        mock_z_values,
        1,
        least_squares_fit_weights=cupy.array([0.85, 1.0, 0.85]),
    )

    mock_radial_profiles = cupy.array([mock_z_lookup_table])
    computed_z = z_lookup_tables.compute_z(mock_radial_profiles)

    assert (computed_z == mock_z_values).all()


def test_multiple_z_lookup_table_nested(
    mock_z_lookup_table: cupy.ndarray, mock_z_values
):
    num_images = 100

    z_lookup_tables = ZLookupTables(
        cupy.array([mock_z_lookup_table] * NUM_Z_LAYERS),
        mock_z_values,
        num_images,
        least_squares_fit_weights=cupy.array([0.85, 1.0, 0.85]),
    )

    mock_radial_profiles = cupy.repeat(
        cupy.expand_dims(mock_z_lookup_table, axis=0), num_images, axis=0
    )

    computed_z_nested = z_lookup_tables.compute_z(mock_radial_profiles)
    assert computed_z_nested.shape[0] == num_images

    for computed_z in computed_z_nested:
        assert (computed_z == mock_z_values).all()
