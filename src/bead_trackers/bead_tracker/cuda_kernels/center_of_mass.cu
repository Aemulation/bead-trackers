using uint8_t = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;

typedef struct RoiCoordinate {
  uint32_t y;
  uint32_t x;
} RoiCoordinate;

typedef struct CenterOfMassCoordinate {
  float y;
  float x;
} CenterOfMassCoordinate;

template <typename I>
__global__ void
calculate_averages(const I *images, const uint32_t num_images,
                   const uint32_t image_height, const uint32_t image_width,
                   const RoiCoordinate *roi_coordinates, size_t num_rois,
                   const uint32_t roi_size, float *roi_averages_per_image,
                   float *roi_stds_per_image) {
  const size_t roi_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (roi_id >= num_rois) {
    return;
  }

  const size_t image_id = blockIdx.y;
  if (image_id >= num_images) {
    return;
  }

  const size_t image_size =
      static_cast<size_t>(image_height) * static_cast<size_t>(image_width);
  const size_t num_roi_pixels =
      static_cast<size_t>(roi_size) * static_cast<size_t>(roi_size);

  float *roi_averages = &roi_averages_per_image[image_id * num_rois];
  float *roi_stds = &roi_stds_per_image[image_id * num_rois];
  const RoiCoordinate roi_coordinate = roi_coordinates[roi_id];
  const I *image = &images[image_id * image_size];

  float total = 0;
  float total_squared = 0;
  for (size_t row_id{}; row_id != roi_size; ++row_id) {
    const size_t image_row_offset = (roi_coordinate.y + row_id) * image_width;

    for (size_t column_id{}; column_id != roi_size; ++column_id) {
      const size_t image_offset =
          image_row_offset + column_id + roi_coordinate.x;

      const I pixel_value = image[image_offset];
      total += pixel_value;
      total_squared +=
          static_cast<float>(pixel_value) * static_cast<float>(pixel_value);
    }
  }

  const float roi_average = total / num_roi_pixels;
  const float roi_std =
      sqrtf(total_squared / num_roi_pixels - roi_average * roi_average);

  roi_averages[roi_id] = roi_average;
  roi_stds[roi_id] = roi_std;
}

template <typename I>
__global__ void
center_of_mass(const I *images, const uint32_t num_images,
               const uint32_t image_height, const uint32_t image_width,
               const RoiCoordinate *roi_coordinates, size_t num_rois,
               const uint32_t roi_size, const float *roi_averages_per_image,
               const float *roi_stds_per_image,
               CenterOfMassCoordinate *center_of_mass_coordinates_per_image) {
  const size_t roi_id = blockDim.x * blockIdx.x + threadIdx.x;
  if (roi_id >= num_rois) {
    return;
  }

  const size_t image_id = blockIdx.y;
  if (image_id >= num_images) {
    return;
  }

  const size_t image_size =
      static_cast<size_t>(image_height) * static_cast<size_t>(image_width);

  CenterOfMassCoordinate *center_of_mass_coordinates =
      &center_of_mass_coordinates_per_image[image_id * num_rois];
  const float roi_average =
      roi_averages_per_image[image_id * num_rois + roi_id];
  const float roi_std = roi_stds_per_image[image_id * num_rois + roi_id];
  const RoiCoordinate roi_coordinate = roi_coordinates[roi_id];
  const I *image = &images[image_id * image_size];

  float total_II = 0;
  float total_y_II = 0;
  float total_x_II = 0;

  // TODO: Make configurable.
  const float correction_factor = 1.0f;

  for (size_t row_id{}; row_id != roi_size; ++row_id) {
    const size_t image_row_offset = (roi_coordinate.y + row_id) * image_width;

    for (size_t column_id{}; column_id != roi_size; ++column_id) {
      const size_t image_offset =
          image_row_offset + column_id + roi_coordinate.x;

      const I pixel_value = image[image_offset];
      const float II =
          fmaxf(0.0f, fabsf(static_cast<float>(pixel_value) - roi_average) -
                          correction_factor * roi_std);

      total_II += II;
      total_y_II += row_id * II;
      total_x_II += column_id * II;
    }
  }

  center_of_mass_coordinates[roi_id] =
      CenterOfMassCoordinate{total_y_II / total_II + roi_coordinate.y,
                             total_x_II / total_II + roi_coordinate.x};
}
