
[[vk::binding(0,0)]]
StructuredBuffer<float> lhs;

[[vk::binding(1,0)]]
StructuredBuffer<float> rhs;

[[vk::binding(2,0)]]
RWStructuredBuffer<float> result;

struct BinaryMatrixOperatorMeta {
    uint2 lhsStride;
    uint2 rhsStride;
    uint2 resultStride;
    uint innerDim;
}

[[vk::binding(3,0)]]
ConstantBuffer<BinaryMatrixOperatorMeta> c;

uint index(uint2 baseIndex, uint2 stride) {
    return dot(baseIndex * stride, uint2(1, 1));
}

[shader("compute")]
[numthreads(1,1,1)] // TODO: Make this work with threads ^^
void matmul(uint3 threadId : SV_DispatchThreadID) {
    uint2 i2 = threadId.xy;
    float sum = 0;
    for (uint i = 0; i < c.innerDim; ++i) {
        sum += lhs[index(uint2(i2.x, i), c.lhsStride)] * rhs[index(uint2(i, i2.y), c.rhsStride)];
    }
    result[index(i2, c.resultStride)] = sum;
}
