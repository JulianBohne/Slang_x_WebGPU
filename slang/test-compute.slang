
[[vk::binding(0,0)]]
RWStructuredBuffer<float> buffer;

[shader("compute")]
[numthreads(1,1,1)] // Note: Work group size of 64 is a good starting point. If our buffer isn't cleanly divisible by 64, use a condition in the shader to not do anything ^^
void computeMain(uint3 threadId : SV_DispatchThreadID) {
    uint i = threadId.x;
    buffer[i] = buffer[i] * buffer[i];
}
