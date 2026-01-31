
[vk::binding(0, 0)]
Texture2D tex;

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

struct VOut {
    float4 position : SV_Position;
    float2 uv;
}

static const float2[] pos = {
    float2(-1., -1.), // bottom left
    float2(1., -1.),  // bottom right
    float2(1., 1.),   // top right
    float2(1., 1.),   // top right
    float2(-1., 1.),  // top left
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
