using uint8_t = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using int64_t = long long int;

__device__ size_t find_min_index(const float *values, size_t num_values) {
    size_t min_index = 0;
    float min_value = values[0];

    for (size_t value_id = 1; value_id != num_values; ++value_id) {
        const float value = values[value_id];

        if (value < min_value) {
            min_index = value_id;
            min_value = value;
        }
    }

    return min_index;
}

__device__ size_t find_max_index(const float *values, size_t num_values) {
    size_t max_index = 0;
    float max_value = values[0];

    for (size_t value_id = 1; value_id != num_values; ++value_id) {
        const float value = values[value_id];

        if (value > max_value) {
            max_index = value_id;
            max_value = value;
        }
    }

    return max_index;
}

__global__ void copy_relevant_points_min(const float *match_scores_table, const uint32_t num_beads,
                                         const uint32_t num_values_per_bead, const size_t num_points_per_bead,
                                         float *points_table, float *starting_points) {
    const size_t bead_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (bead_id >= num_beads) {
        return;
    }

    const float *match_scores = &match_scores_table[bead_id * num_values_per_bead];

    const size_t min_index = find_min_index(match_scores, num_values_per_bead);

    const uint32_t start_position =
        max(static_cast<long long int>(0),
            static_cast<long long int>(min_index) - static_cast<long long int>(num_points_per_bead / 2));
    const uint32_t end_position = min(static_cast<long long int>(num_values_per_bead - 1),
                                      static_cast<long long int>(min_index + num_points_per_bead / 2));

    if (end_position - start_position + 1 < num_points_per_bead) {
        starting_points[bead_id] = start_position;
        for (size_t point_id = 0; point_id != num_points_per_bead; ++point_id) {
            const uint32_t distance = abs(static_cast<int64_t>(point_id + start_position - min_index));
            points_table[bead_id * num_points_per_bead + point_id] = distance * distance + distance;
        }
    } else {
        starting_points[bead_id] = start_position;
        for (size_t point_id = 0; point_id != num_points_per_bead; ++point_id) {
            points_table[bead_id * num_points_per_bead + point_id] = match_scores[start_position + point_id];
        }
    }
}

__global__ void copy_relevant_points_max(const float *match_scores_table, const uint32_t num_beads,
                                         const uint32_t num_values_per_bead, const size_t num_points_per_bead,
                                         float *points_table, float *starting_points) {
    const size_t bead_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (bead_id >= num_beads) {
        return;
    }

    const float *match_scores = &match_scores_table[bead_id * num_values_per_bead];

    const size_t max_index = find_max_index(match_scores, num_values_per_bead);

    const uint32_t start_position =
        max(static_cast<long long int>(0),
            static_cast<long long int>(max_index) - static_cast<long long int>(num_points_per_bead / 2));
    const uint32_t end_position = min(static_cast<long long int>(num_values_per_bead - 1),
                                      static_cast<long long int>(max_index + num_points_per_bead / 2));

    if (end_position - start_position + 1 < num_points_per_bead) {
        starting_points[bead_id] = start_position;
        for (size_t point_id = 0; point_id != num_points_per_bead; ++point_id) {
            const uint32_t distance = abs(static_cast<int64_t>(point_id + start_position - max_index));
            points_table[bead_id * num_points_per_bead + point_id] = -(distance * distance + distance);
        }
    } else {
        starting_points[bead_id] = start_position;
        for (size_t point_id = 0; point_id != num_points_per_bead; ++point_id) {
            points_table[bead_id * num_points_per_bead + point_id] = match_scores[start_position + point_id];
        }
    }
}
