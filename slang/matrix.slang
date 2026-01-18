
struct Matrix {
    uint2 shape;
    uint2 stride;
    StructuredBuffer<float> buf;

    uint index(uint2 baseIndex) {
        return dot(baseIndex * stride, uint2(1, 1));
    }

    float get(uint2 baseIndex) {
        return buf[index(baseIndex)];
    }
}

struct RWMatrix {
    uint2 shape;
    uint2 stride;
    RWStructuredBuffer<float> buf;

    uint index(uint2 baseIndex) {
        return dot(baseIndex * stride, uint2(1, 1));
    }

    float get(uint2 baseIndex) {
        return buf[index(baseIndex)];
    }

    void set(uint2 baseIndex, float value) {
        buf[index(baseIndex)] = value;
    }
}
[[vk::binding(0,0)]]
ConstantBuffer<RWMatrix> result;

[[vk::binding(0,1)]]
ConstantBuffer<Matrix> lhs;

// [[vk::binding(0,1)]]
// ConstantBuffer<float> value_to_set;

[[vk::binding(0,2)]]
ConstantBuffer<Matrix> rhs;

[shader("compute")]
[numthreads(1,1,1)] // TODO: Make this work with threads ^^
void set_zero(uint3 threadId : SV_DispatchThreadID) {
    uint2 i2 = threadId.xy;
    result.set(i2, 0);
}

[shader("compute")]
[numthreads(1,1,1)] // TODO: Make this work with threads ^^
void set_one(uint3 threadId : SV_DispatchThreadID) {
    uint2 i2 = threadId.xy;
    result.set(i2, 1);
}

[shader("compute")]
[numthreads(1,1,1)] // TODO: Make this work with threads ^^
void matmul_forwards(uint3 threadId : SV_DispatchThreadID) {
    uint2 i2 = threadId.xy;
    float sum = 0;
    for (uint i = 0; i < lhs.shape.y; ++i) {
        sum += lhs.get(uint2(i2.x, i)) * rhs.get(uint2(i, i2.y));
    }
    result.set(i2, sum);
}

[shader("compute")]
[numthreads(1,1,1)] // TODO: Make this work with threads ^^
void matmul_backwards(uint3 threadId : SV_DispatchThreadID) {
    uint2 i2 = threadId.xy;
    float sum = 0;
    for (uint i = 0; i < lhs.shape.y; ++i) {
        sum += lhs.get(uint2(i2.x, i)) * rhs.get(uint2(i, i2.y));
    }
    result.buf[result.index(i2)] += sum;
}
