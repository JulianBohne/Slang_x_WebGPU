module gaussian;

public float cool_func(float2 uv) {
    return length(uv - float2(0.5)) > 0.5 ? 0 : 1;
}

[Differentiable]
public float gaussianPDF(float2 coord, float2 mew, float2x2 invCov) {
    return exp(-0.5 * dot((coord - mew), mul(invCov, coord - mew)));
}

struct SplatCamera {
    float4x4 mv;
    float4x4 projection;
    float4x4 mvp;

    float3x3 jacobian(float3 at) {
        // TODO: Double chjeck this lol
        float a = projection[0][0];
        float b = projection[1][1];
        float d = projection[3][2];

        return transpose(float3x3(
            -a/at.z,       0, (a * at.x) / (at.z * at.z),
                  0, -b/at.z, (b * at.y) / (at.z * at.z),
                  0,       0,          d / (at.z * at.z),
        ));
    }

}

struct Gaussian2D {
    float2 position;
    float2x2 inv_cov;
    float3 color;
    float alpha;
}

struct Gaussian2DHelpers {
    float2 axis_a;
    float2 axis_b;
    float depth;
}

struct TransformedGaussian {
    Gaussian2D gaussian;
    Gaussian2DHelpers helpers;
}

struct Gaussian3D : IDifferentiable {
    float3 position;
    float4 rotation;
    float3 scale;
    float3 color;
    float alpha;

    [Differentiable]
    TransformedGaussian transform(SplatCamera cam) {
        cam.jacobian(this.position);
    }
}

static const int MAX_GAUSSIANS_PER_TILE = 256;
static const int TILE_SIZE = 16;

[[vk::binding(0,1)]]
RWStructuredBuffer<Gaussian3D> gaussians_3d;

[[vk::binding(0,2)]]
RWStructuredBuffer<Gaussian2D> gaussians_2d;

// WGSL doen't really support atomics for much, so we gotta bitcast to uints :D
[[vk::binding(0,3)]]
RWStructuredBuffer<Atomic<uint>> grad_buffer;

// One workgroup is one tile

// Number of gaussians in this tile
groupshared Atomic<uint> atomic_gaussian_count;
groupshared uint gaussian_count;

groupshared uint gaussians[MAX_GAUSSIANS_PER_TILE];
