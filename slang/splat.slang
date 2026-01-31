import gaussian;

static const uint SENTINAL = -1; // 0xffffffff;
static const uint MAX_GAUSSIANS_PER_TILE = 256*8;
static const uint TILE_SIZE = 16;

[[vk::binding(0, 0)]]
ConstantBuffer<SplatCamera> camera;

[[vk::binding(1, 0)]]
RWStructuredBuffer<Gaussian3D> gaussians_3d;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Gaussian2D> gaussians_2d;

[[vk::binding(3, 0)]]
RWStructuredBuffer<float> gaussian_depths;

[[vk::binding(4, 0)]]
RWStructuredBuffer<uint> sorted_indices;

[[vk::binding(5, 0)]]
WTexture2D output;

// // WGSL doen't really support atomics for much, so we gotta bitcast to uints :D
// [[vk::binding(2, 1)]]
// RWStructuredBuffer<Atomic<uint>> grad_buffer;

// One workgroup is one tile

// Number of gaussians in this tile
groupshared Atomic<uint> atomic_num_gaussians_in_tile;
groupshared uint num_gaussians_in_tile;

groupshared uint gaussians[MAX_GAUSSIANS_PER_TILE];

void collect_gaussians_in_tile(uint num_gaussians, uint groupThreadIndex, float2 tile_ndc_min, float2 tile_ndc_max) {
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIndex == 0) {
        atomic_num_gaussians_in_tile = 0;
    }
    GroupMemoryBarrierWithGroupSync(); // Not sure if this second barrier is necessary? like, should this not run in lockstep?

    float2 quad_points[4] = {
        tile_ndc_min,
        float2(tile_ndc_min.x, tile_ndc_max.y),
        tile_ndc_max,
        float2(tile_ndc_max.x, tile_ndc_min.y),
    };

    // TODO: Better gaussian in tile check :)
    for (uint j = groupThreadIndex; j < num_gaussians; j += TILE_SIZE * TILE_SIZE /* work group size */) {
        uint i = sorted_indices[j];
        if (gaussians_2d[i].depth >= 0 && gaussians_2d[i].depth <= 1 && gaussians_2d[i].intersects_quad(quad_points)) {
            uint index = atomic_num_gaussians_in_tile++;
            if (index >= MAX_GAUSSIANS_PER_TILE) break;
            gaussians[index] = j; // Index in sorted index buffer
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIndex == 0) {
        num_gaussians_in_tile = min(min(atomic_num_gaussians_in_tile.load(), MAX_GAUSSIANS_PER_TILE), num_gaussians);
    }
    GroupMemoryBarrierWithGroupSync(); // Not sure if this second barrier is necessary? like, should this not run in lockstep?
}

void pad_buffer_with_sentinal(uint groupThreadIndex) {
    for (uint i = num_gaussians_in_tile + groupThreadIndex; i < MAX_GAUSSIANS_PER_TILE; i += TILE_SIZE * TILE_SIZE) {
        gaussians[i] = uint::maxValue;
    }
}

// Bitonic sort stolen from https://shader-slang.org/slang-playground/?target=SPIRV&code=eJyVU01r20AQvetXDD7Z4MqJm0OJUh_iXHoIDdgEijFlI41tkdWu2J11bUr_e2c_vJHpqUKg1czozXszT7MZrA-tBTyJrpcIDXZaWTKC0MJB_wLSHNu1CuGIhvAEQjWwM2LfoSKwB9GgAT6ac69bRbYsZgmSbwG17vpW4iet5PnSZApWAx3QoK9RGrSj3hGj6s7HoZfivDfaqSag_dAOaqHAWQzp0TKCjuDNEWnlOaY-nB-I8dzb3Y4bMVUSZo-eX8HyXE3wGvQUvwvgaye1oM_Qa9tSq1U1DCptOiEHoTm4Y1X8yUjPPC3TCpmweAwg5Bs2eo0ncga_qQZPVc5FvOvcB9iqRoUJaRVCXNU8Oq_j4dJpAV062Yib0OZPQPFkN7fzL9sh8FJ0aMRQ793pDrpjH4pehOE8Yz5KXb8_BBYLsP5RFUvvCaEosYhIC16Kf1bFKozbrIg5gR28VB_D_u7oqnUeNdzD6vXny_8NfhONNx5FU44m28J3SB59Fq0ax_XywNlak9Q61ES3RdR4LjOXr9A5OY66Sh7NNLEdB5hcN4Xb8mYyucKIZBkhlg65pwp3zFkvw2cM8q5UpjRUdvnHvLY0skso6vNaoro4wnUweFIqkS5O4KZhjWV2RnzNDtrcbMt__bqtMlCtpTYMkwDKuO_xcNPTLGxypSx86oX9Be1nf3g
/*
 * bitonicSort() implements a workgroup-level parallel sorting algorithm to sort indices in the short-list.
 * Requires all elements in the buffer to be valid (invalid elements should be set to infinity, or its equivalent).
 *
 * Bitonic sorting is an efficient, deterministic, parallel sorting algorithm particularly well-suited for GPUs.
 * At a high-level, it operates by comparing & swapping elements in parallel in (logN)^2 stages.
 *
 * More info on the bitonic sort algorithm: https://en.wikipedia.org/wiki/Bitonic_sorter
 * The code was adapted from the Wikipedia sample pseudocode here: https://en.wikipedia.org/wiki/Bitonic_sorter#Example_code
 *
 */
void sort_gaussians_in_tile(uint groupThreadIndex) {
    pad_buffer_with_sentinal(groupThreadIndex);

    GroupMemoryBarrierWithGroupSync();
    for (uint k = 2; k <= MAX_GAUSSIANS_PER_TILE; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            for (uint i = groupThreadIndex; i < MAX_GAUSSIANS_PER_TILE; i += TILE_SIZE * TILE_SIZE) {
                uint l = i ^ j;
                if (l > i)
                {
                    if ((((i & k) == 0) && (gaussians[i] > gaussians[l])) ||
                        (((i & k) != 0) && (gaussians[i] < gaussians[l])))
                    {
                        // Swap
                        uint tmp = gaussians[i];
                        gaussians[i] = gaussians[l];
                        gaussians[l] = tmp;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}

void translate_indices(uint groupThreadIndex) {
    GroupMemoryBarrierWithGroupSync();
    for (uint i = groupThreadIndex; i < num_gaussians_in_tile; i += TILE_SIZE * TILE_SIZE) {
        gaussians[i] = sorted_indices[gaussians[i]];
    }
    GroupMemoryBarrierWithGroupSync();
}

float sigmoid(float x) {
    return 1.0 / (1 + exp(-x));
}

[shader("compute")]
[NumThreads(256, 1, 1)]
void transform_gaussians(uint3 threadId: SV_DispatchThreadID) {
    uint global_thread_id = threadId.x;

    uint num_total_gaussians, gaussian_size;
    gaussians_3d.GetDimensions(num_total_gaussians, gaussian_size);

    if (global_thread_id < num_total_gaussians) {
        gaussians_2d[global_thread_id] = gaussians_3d[global_thread_id].transform(camera);
        float base_depth = gaussians_2d[global_thread_id].depth;
        gaussian_depths[global_thread_id] = sigmoid(base_depth);
        sorted_indices[global_thread_id] = global_thread_id;
    }
}

[shader("compute")]
[NumThreads(TILE_SIZE, TILE_SIZE, 1)]
void render(uint3 threadId: SV_DispatchThreadID, uint3 groupThreadId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    uint output_width;
    uint output_height;
    output.GetDimensions(output_width, output_height);

    uint num_total_gaussians, gaussian_2d_size;
    gaussians_2d.GetDimensions(num_total_gaussians, gaussian_2d_size);

    uint tiles_count_x = ((output_width - 1) / TILE_SIZE) + 1;
    uint tiles_count_y = ((output_width - 1) / TILE_SIZE) + 1;

    uint global_thread_id = threadId.x + threadId.y * (tiles_count_x * TILE_SIZE);
    uint num_threads = TILE_SIZE * TILE_SIZE * tiles_count_x * tiles_count_y;

    float2 tile_uv_min = float2(groupId.xy * TILE_SIZE) / float2(output_width, output_height);
    float2 tile_uv_max = min(float2((groupId.xy + 1) * TILE_SIZE) / float2(output_width, output_height), float2(1, 1));

    uint groupThreadIndex = groupThreadId.x + groupThreadId.y * TILE_SIZE;

    collect_gaussians_in_tile(num_total_gaussians, groupThreadIndex, (tile_uv_min * 2 - 1) * float2(1, -1), (tile_uv_max * 2 - 1) * float2(1, -1));
    
    sort_gaussians_in_tile(groupThreadIndex);

    translate_indices(groupThreadIndex);

    if (!all(threadId.xy < uint2(output_width, output_height))) {
        return; // Outside of range
    }

    float2 pixel_ndc = (2 * (float2(threadId.xy) / float2(output_width, output_height)) - 1) * float2(1, -1);

    float opacity = 0.0;
    float3 color = float3(0);
    [MaxIters(MAX_GAUSSIANS_PER_TILE)]
    for (uint i = 0; i < num_gaussians_in_tile; ++i) {
        uint index = gaussians[i];
        float density = gaussianPDF(pixel_ndc, gaussians_2d[index].position, gaussians_2d[index].inv_cov);
        float t = (1 - opacity) * gaussians_2d[index].alpha * density;
        color = lerp(color, gaussians_2d[index].color, t);
        opacity = lerp(opacity, 1, t);
        if (opacity > 0.99) break;
    }

    // output.Store(threadId.xy, float4(float3(gaussians_2d[0].inv_cov[0][0]), 1));
    output.Store(threadId.xy, num_gaussians_in_tile == MAX_GAUSSIANS_PER_TILE ? lerp(float4(color, 1), float4(1, 0, 1, 1), 0.5) : float4(color, 1));
}
