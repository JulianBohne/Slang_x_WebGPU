
[[vk::binding(0,0)]]
RWStructuredBuffer<float> buffer;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID) {
    uint i = threadId.x;
    buffer[i] = buffer[i] * buffer[i];
}
