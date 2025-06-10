using uint8_t = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;

typedef struct RoiCoordinate {
  uint32_t y;
  uint32_t x;
} RoiCoordinate;

typedef struct BeadCoordinate {
  float y;
  float x;
} BeadCoordinate;

template <typename I>
__device__ float interpolate(const I *image, const uint32_t height,
                             const uint32_t width, const float y,
                             const float x) {
  const size_t y_floored = floorf(y);
  const size_t x_floored = floorf(x);
  const size_t y_ceiled = ceilf(y);
  const size_t x_ceiled = ceilf(x);

  const float y_fraction = y - y_floored;
  const float x_fraction = x - x_floored;

  const float pixel_value_00 = image[(y_floored * width + x_floored)];
  const float pixel_value_01 = image[(y_floored * width + x_ceiled)];
  const float pixel_value_10 = image[(y_ceiled * width + x_floored)];
  const float pixel_value_11 = image[(y_ceiled * width + x_ceiled)];

  const float interpolated_y0_value =
      pixel_value_00 + x_fraction * (pixel_value_01 - pixel_value_00);
  const float interpolated_y1_value =
      pixel_value_10 + x_fraction * (pixel_value_11 - pixel_value_10);

  const float interpolated_value =
      interpolated_y0_value +
      y_fraction * (interpolated_y1_value - interpolated_y0_value);

  return interpolated_value;
}

template <typename I>
__global__ void radial_profile(
    const I *images, const uint32_t num_images, const uint32_t image_height,
    const uint32_t image_width, const RoiCoordinate *roi_coordinates,
    const uint32_t roi_size, const BeadCoordinate *bead_positions_per_image,
    const size_t num_beads, const float *radials, const size_t num_radials,
    const BeadCoordinate *radial_positions, const size_t num_radial_positions,
    const float *averages, float *radial_profiles_per_image) {
  const size_t bead_id = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t radial_id = blockDim.y * blockIdx.y + threadIdx.y;
  const size_t image_id = blockIdx.z;

  if (bead_id >= num_beads || radial_id >= num_radials ||
      image_id >= num_images) {
    return;
  }

  const BeadCoordinate *bead_positions =
      &bead_positions_per_image[image_id * num_beads];
  const size_t image_size =
      static_cast<size_t>(image_height) * static_cast<size_t>(image_width);
  const I *image = &images[image_id * image_size];
  float *radial_profiles =
      &radial_profiles_per_image[image_id * num_radials * num_beads];
  const uint32_t roi_y = roi_coordinates[bead_id].y;
  const uint32_t roi_x = roi_coordinates[bead_id].x;

  const float bead_y_position = bead_positions[bead_id].y;
  const float bead_x_position = bead_positions[bead_id].x;

  const float radial = radials[radial_id];

  float summed_interpolated_pixel_value = 0;
  size_t valid_radial_steps = 0;

  for (size_t radial_position_step = 0;
       radial_position_step != num_radial_positions; ++radial_position_step) {
    const float y =
        bead_y_position + radial_positions[radial_position_step].y * radial;
    const float x =
        bead_x_position + radial_positions[radial_position_step].x * radial;

    if (y < roi_y || y >= roi_y + roi_size || x < roi_x ||
        x >= roi_x + roi_size) {
      continue;
    }

    summed_interpolated_pixel_value +=
        interpolate<I>(image, image_height, image_width, y, x);
    ++valid_radial_steps;
  }

  if (valid_radial_steps <= 4) {
    radial_profiles[bead_id * num_radials + radial_id] =
        averages[image_id * num_beads + bead_id];
  } else {
    radial_profiles[bead_id * num_radials + radial_id] =
        summed_interpolated_pixel_value / valid_radial_steps;
  }
}
