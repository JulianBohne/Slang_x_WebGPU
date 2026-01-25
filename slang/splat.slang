import gaussian;

static const uint SENTINAL = -1; // 0xffffffff;
static const uint MAX_GAUSSIANS_PER_TILE = 256;
static const uint TILE_SIZE = 16;

[[vk::binding(0, 0)]]
ConstantBuffer<SplatCamera> camera;

[[vk::binding(1, 0)]]
RWStructuredBuffer<Gaussian3D> gaussians_3d;

[[vk::binding(2, 0)]]
RWStructuredBuffer<Gaussian2D> gaussians_2d;

[[vk::binding(3, 0)]]
WTexture2D output;

// // WGSL doen't really support atomics for much, so we gotta bitcast to uints :D
// [[vk::binding(2, 1)]]
// RWStructuredBuffer<Atomic<uint>> grad_buffer;

// One workgroup is one tile

// Number of gaussians in this tile
groupshared Atomic<uint> atomic_num_gaussians_in_tile;
groupshared uint num_gaussians_in_tile;

groupshared uint gaussians[MAX_GAUSSIANS_PER_TILE];

void transform_gaussians(uint thread_index, uint num_threads, uint num_gaussians) {
    for (uint i = thread_index; i < num_gaussians; i += num_threads) {
        gaussians_2d[i] = gaussians_3d[i].transform(camera);
    }
}

void collect_gaussians_in_tile(uint num_gaussians, uint3 groupThreadId, uint3 groupId) {
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadId.x == 0 && groupThreadId.y == 0) {
        atomic_num_gaussians_in_tile = 0;
    }
    GroupMemoryBarrierWithGroupSync(); // Not sure if this second barrier is necessary? like, should this not run in lockstep?

    // TODO: Check if gaussian is in Tile
    // Currently we're just saying YES, for future reference, this is where we do wacky transformations with the axes :)
    for (uint j = groupThreadId.x + groupThreadId.y * TILE_SIZE; j < num_gaussians; j += MAX_GAUSSIANS_PER_TILE /* work group size */) {
        if (gaussians_2d[j].depth >= 0 && gaussians_2d[j].depth <= 1) {
            uint index = atomic_num_gaussians_in_tile++;
            if (index >= MAX_GAUSSIANS_PER_TILE) break;
            gaussians[index] = j;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadId.x == 0) {
        num_gaussians_in_tile = min(min(atomic_num_gaussians_in_tile.load(), MAX_GAUSSIANS_PER_TILE), num_gaussians);
    }
    GroupMemoryBarrierWithGroupSync(); // Not sure if this second barrier is necessary? like, should this not run in lockstep?
}

[shader("compute")]
[NumThreads(TILE_SIZE, TILE_SIZE, 1)]
void render(uint3 threadId: SV_DispatchThreadID, uint3 groupThreadId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    uint output_width;
    uint output_height;
    output.GetDimensions(output_width, output_height);

    uint num_total_gaussians, gaussian_size;
    gaussians_3d.GetDimensions(num_total_gaussians, gaussian_size);

    uint tiles_count_x = ((output_width - 1) / TILE_SIZE) + 1;
    uint tiles_count_y = ((output_width - 1) / TILE_SIZE) + 1;

    uint global_thread_id = threadId.x + threadId.y * (tiles_count_x * TILE_SIZE);
    uint num_threads = TILE_SIZE * TILE_SIZE * tiles_count_x * tiles_count_y;

    // NOTE: For now I will do the transformation here, but this couples the number of dispatched threads for transforming the gaussians to the resolution of the render, which is dumb lol
    transform_gaussians(global_thread_id, num_threads, num_total_gaussians);

    collect_gaussians_in_tile(num_total_gaussians, groupThreadId, groupId);
    
    // TODO: Sort gaussians (we're just not gonna do that now)

    if (!all(threadId.xy < uint2(output_width, output_height))) {
        return; // Outside of range
    }

    float2 pixel_ndc = 2 * (float2(threadId.xy) / float2(output_width, output_height)) - 1;

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
    output.Store(threadId.xy, float4(color, 1));
}

uniform float w7h;
uniform float time;

struct VOut {
    float4 position : SV_Position;
    float2 uv;
}

static const float2[] pos = {
    float2(-1., -1.), // bottom left
    float2( 1., -1.), // bottom right
    float2( 1.,  1.), // top right
    float2( 1.,  1.), // top right
    float2(-1.,  1.), // top left
    float2(-1., -1.), // bottom left
};

static const float2[] uvs = {
    float2(0., 1.), // bottom left
    float2(1., 1.), // bottom right
    float2(1., 0.), // top right
    float2(1., 0.), // top right
    float2(0., 0.), // top left
    float2(0., 1.), // bottom left
};

[vk::binding(0, 0)]
Texture2D tex;

// [vk::binding(1, 0)]
// SamplerState sampler_state;

[shader("vertex")]
VOut vert(uint vertexIndex : SV_VertexID) {
    return VOut(
        float4(pos[vertexIndex], 0.0, 1.0),
        uvs[vertexIndex],
    );
}

[shader("fragment")]
float4 frag(VOut input) : SV_Target {
    // float2 uv = (input.uv - float2(0, 0.5)) / float2(1, w7h) + float2(0, 0.5);
    uint width, height;
    tex.GetDimensions(width, height);
    return tex.Load(uint3(uint(input.uv.x * (width-1)), uint(input.uv.y * (height-1)), 0));
    // return float4(input.uv * (sin(time) * 0.5 + 0.5), 0.0, 1.0);
}
