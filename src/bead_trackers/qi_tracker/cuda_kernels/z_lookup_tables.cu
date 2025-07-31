using uint8_t = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using int64_t = long long int;

__global__ void compute_profile_match_scores(const float *radial_profiles_table_per_image, const float *z_lookup_tables,
                                             const uint32_t num_beads, const uint32_t num_z_planes,
                                             const uint32_t num_radials, float *match_scores_table_per_image) {
    const size_t bead_id = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t z_plane_id = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t image_id = blockIdx.z;

    if (bead_id >= num_beads || z_plane_id >= num_z_planes) {
        return;
    }

    const float *radial_profiles_table = &radial_profiles_table_per_image[image_id * num_beads * num_radials];
    float *match_scores_table = &match_scores_table_per_image[image_id * num_beads * num_z_planes];

    const float *radial_profile = &radial_profiles_table[bead_id * num_radials];
    const float *z_lookup_table = &z_lookup_tables[bead_id * num_radials * num_z_planes];

    float total_radial_difference = 0;
    for (uint32_t radial_id = 0; radial_id != num_radials; ++radial_id) {
        const float radial_difference =
            radial_profile[radial_id] - z_lookup_table[z_plane_id * num_radials + radial_id];

        // TODO: z_cmp_window?
        total_radial_difference += radial_difference * radial_difference;
    }

    match_scores_table[bead_id * num_z_planes + z_plane_id] = total_radial_difference;
}
