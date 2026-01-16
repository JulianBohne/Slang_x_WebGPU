
[[vk::binding(0,0)]]
RWStructuredBuffer<float> buffer;

[Differentiable]
float square(float x) {
    return x * x;
}

[shader("compute")]
[numthreads(1,1,1)] // Note: Work group size of 64 is a good starting point. If our buffer isn't cleanly divisible by 64, use a condition in the shader to not do anything ^^
void computeMain(uint3 threadId : SV_DispatchThreadID) {
    uint i = threadId.x;
    DifferentialPair<float> input = diffPair(buffer[i]);
    float d_output = 1.0;
    bwd_diff(square)(input, d_output);
    buffer[i] = input.d;
}
