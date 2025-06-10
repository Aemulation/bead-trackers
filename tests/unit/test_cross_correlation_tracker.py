# import cupy
# import matplotlib
# import matplotlib.patches
# import matplotlib.pyplot as plt
#
#
# from src.cross_correlation_tracker import CrossCorrelationTracker
#
#
# def visualize_yx_tracker(image: cupy.ndarray, coordinates):
#     fig, ax = plt.subplots()
#     ax.imshow(image.get())
#     circle = matplotlib.patches.Circle(
#         (
#             coordinates[1].get(),
#             coordinates[0].get(),
#         ),
#         radius=2,
#         edgecolor="red",
#         facecolor="red",
#         linewidth=2,
#     )
#     ax.add_patch(circle)
#     plt.show()
#
#
# def test_cross_correlation_tracker_full_image(rings: cupy.ndarray):
#     [image_height, image_width] = rings.shape
#
#     mock_cross_correlation_tracker = CrossCorrelationTracker(
#         1, image_height, image_width, cupy.array([[0, 0]], dtype=cupy.uint32)
#     )
#     bead_coordinates = mock_cross_correlation_tracker.calculate_yx(
#         cupy.expand_dims(rings, axis=0),
#     )
#
#     print("YX YX YX")
#     print(bead_coordinates.shape)
#     print(bead_coordinates)
#
#     visualize_yx_tracker(rings, bead_coordinates[0, 0])
#
#     assert bead_coordinates.shape[0] == 1
#     assert bead_coordinates.shape[1] == 1
#     [bead_coordinate_y, bead_coordinate_x] = bead_coordinates[0][0]
#     assert cupy.isclose(bead_coordinate_y, image_height / 2, atol=1)
#     assert cupy.isclose(bead_coordinate_x, image_width / 2, atol=1)
#
#
# def test_calculate_yx_full_image_real(roi_image: cupy.ndarray):
#     [image_height, image_width] = roi_image.shape
#
#     mock_cross_correlation_tracker = CrossCorrelationTracker(
#         1, image_height, image_width, cupy.array([[0, 0]], dtype=cupy.uint32)
#     )
#     bead_coordinates = mock_cross_correlation_tracker.calculate_yx(
#         cupy.expand_dims(roi_image, axis=0),
#     )
#     assert bead_coordinates.shape[0] == 1
#     assert bead_coordinates.shape[1] == 1
#     [bead_coordinate_y, bead_coordinate_x] = bead_coordinates[0][0]
#     print()
#     print("COORDS")
#     print(bead_coordinate_y)
#     print(bead_coordinate_x)
#
#     visualize_yx_tracker(roi_image, bead_coordinates[0, 0])
#
#     expected_yx = [22.1, 23.5]
#     assert cupy.isclose(bead_coordinate_y, expected_yx[0], atol=0.2)
#     assert cupy.isclose(bead_coordinate_x, expected_yx[1], atol=0.2)
#
#
# # def test_calculate_yx_full_image_real_nested(roi_image: cupy.ndarray):
# #     [image_height, image_width] = roi_image.shape
# #
# #     num_rois = 1
# #     num_images = 100
# #     roi_images = cupy.repeat(roi_image, num_images, axis=0).reshape(
# #         (num_images, image_height, image_width)
# #     )
# #
# #     mock_center_of_mass = CenterOfMass(num_images, num_rois, image_height, image_width)
# #     center_of_masses_nested = mock_center_of_mass.calculate_yx(
# #         roi_images, cupy.array([[0, 0]], dtype=cupy.uint32)
# #     )
# #
# #     assert center_of_masses_nested.shape == (num_images, num_rois, 2)
# #
# #     for center_of_masses in center_of_masses_nested:
# #         assert center_of_masses.shape[0] == num_rois
# #         [center_of_mass_y, center_of_mass_x] = center_of_masses[0]
# #
# #         expected_yx = [36.9, 52.8]
# #         assert cupy.isclose(center_of_mass_y, expected_yx[0], atol=0.2 * image_height)
# #         assert cupy.isclose(center_of_mass_x, expected_yx[1], atol=0.2 * image_width)
