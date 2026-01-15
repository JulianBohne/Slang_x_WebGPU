
function fail() {
    console.error('WebGPU is not enabled / available :(');
    throw new Error('WebGPU is not enabled / available :(');
}

if (!('gpu' in navigator)) fail();

const adapter = await navigator.gpu.requestAdapter({
    featureLevel: 'core'
});
if (!adapter) fail();

const device = await adapter!.requestDevice();

const shaderCode = await fetch('/wgsl/compiled-test-compute.wgsl').then(res => res.text());

const module = device.createShaderModule({
    label: 'Compute test shader',
    code: shaderCode,
});

const pipeline = device.createComputePipeline({
    label: 'Test compute pipeline',
    layout: "auto",
    compute: {
        module,
    },
});

// Done: Figure out if we could also have written this data to the GPU with a MAP_WRITE buffer :D
// Result: Yes, this should be possible afaik

// Create and write data into GPU buffer
const input = new Float32Array([ 1, 2, 6, 42 ]);
const inputBuffer = device.createBuffer({
    label: 'Input buffer for our compute shader <3',
    size: input.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(inputBuffer, 0, input);

// Create a buffer for mapping results from GPU to JS
const outputBuffer = device.createBuffer({
    label: 'Output buffer for reading back data from the GPU',
    size: input.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
});

const bindGroup = device.createBindGroup({
    label: 'I am group.',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: inputBuffer, // NOTE: The following is also possible - `resource: { buffer: inputBuffer }`
        },
    ],
});

const encoder = device.createCommandEncoder({ label: 'Cool encode', });
const pass = encoder.beginComputePass({ label: 'Our compute pass' });
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(input.length);
pass.end();
encoder.copyBufferToBuffer(inputBuffer, outputBuffer);
const commands = encoder.finish();

device.queue.submit([ commands ]);

await outputBuffer.mapAsync(GPUMapMode.READ);
const result = new Float32Array(outputBuffer.getMappedRange());

console.log('input', input);
console.log('result', result);

outputBuffer.unmap(); // This is important to make the contents available for the GPU again!

