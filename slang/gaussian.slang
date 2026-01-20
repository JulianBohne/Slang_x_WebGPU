module gaussian;

public float cool_func(float2 uv) {
    return length(uv - float2(0.5)) > 0.5 ? 0 : 1;
}

[Differentiable]
public float gaussianPDF(float2 coord, float2 mew, float2x2 invCov) {
    return exp(-0.5 * dot((coord - mew), mul(invCov, coord - mew)));
}

struct Gaussian {
    float4 position;
    float4 rotation;
    float3 scale;
    float4 color;

}
