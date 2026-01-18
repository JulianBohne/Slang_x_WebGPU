
var device: GPUDevice | undefined = undefined;
var encoder: GPUCommandEncoder | undefined = undefined;
var pass: GPUComputePassEncoder | undefined = undefined;

var resolveSubmit: () => void = () => {};
var submitted: Promise<void> | undefined = undefined;


export async function Init(): Promise<boolean> {
  if (!('gpu' in navigator)) return false;

  const adapter = await navigator.gpu.requestAdapter({
    featureLevel: 'core'
  });
  if (!adapter) return false;

  device = await adapter!.requestDevice();

  encoder = device.createCommandEncoder({ label: 'Compute Command Encoder' });

  console.log(adapter);
  console.log(device);

  if (!await Matrix.Init()) return false;

  return true;
}

function AssertInPass() {
  if (pass === undefined) throw new Error('Expected to have an active compute pass encoder. Forgor to call BeginPass?');
}

function AssertNotInPass() {
  if (pass !== undefined) throw new Error('Expected not to have an active compute pass encoder. Forgor to call EndPass?');
}

export function BeginPass() {
  pass = encoder!.beginComputePass({ label: 'Compute Pass Encoder' });
  submitted = new Promise<void>(resolve => { resolveSubmit = () => resolve() });
}

export function EndPass() {
  pass!.end();
  pass = undefined;
}

export function Submit() {
  device!.queue.submit([ encoder!.finish() ]);
  resolveSubmit();
}

export class Matrix {

  shape: [number, number];
  stride: [number, number];

  buffer: GPUBuffer;
  gradBuffer: GPUBuffer | undefined;
  readBuffer: GPUBuffer | undefined;

  requires_grad: boolean;

  backwards: () => void;

  constructor(other: Matrix);
  constructor(shape: [number, number], requires_grad: boolean)
  constructor(arg: [number, number] | Matrix, requires_grad?: boolean) {
    if (arg instanceof Array) {
      const shape = arg;
      const [rows, cols] = shape;
      this.shape  = [rows, cols];
      this.stride = [cols, 1];

      this.buffer = device!.createBuffer({
        label: 'Matrix Buffer',
        size: 4 * rows * cols,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

      this.requires_grad = requires_grad ?? true;

      if (this.requires_grad) {
        this.gradBuffer = device!.createBuffer({
          label: 'Gradient Buffer',
          size: 4 * rows * cols,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
      }

      this.backwards = () => {};

      this.readBuffer = undefined;
    } else {
      const other = arg;
      this.shape = other.shape;
      this.stride = other.stride;
      this.buffer = other.buffer;
      this.gradBuffer = other.gradBuffer;
      this.readBuffer = other.readBuffer;
      this.requires_grad = other.requires_grad;
      this.backwards = other.backwards;
    }
  }

  transpose() {
    const result = new Matrix(this);
    result.shape = [this.shape[1], this.shape[0]];
    result.stride = [this.stride[1], this.stride[0]];
    return result;
  }

  grads() {
    if (!this.requires_grad) throw new TypeError('Can\'t convert matrix that doesn\'t requier_grad to gradient target');
    const result = new Matrix(this);
    const tmp = result.gradBuffer!;
    result.gradBuffer = result.buffer;
    result.buffer = tmp;
    return result;
  }

  private static createMultBindGroup(lhs: Matrix, rhs: Matrix, result: Matrix) {
    // https://webgpufundamentals.org/webgpu/lessons/resources/wgsl-offset-computer.html
    const metaDataBuffer = new ArrayBuffer(32);
    const lhsStride      = new Uint32Array(metaDataBuffer,  0, 2);
    const rhsStride      = new Uint32Array(metaDataBuffer,  8, 2);
    const resultStride   = new Uint32Array(metaDataBuffer, 16, 2);
    const innerDim       = new Uint32Array(metaDataBuffer, 24, 1);

    lhsStride.set(lhs.stride);
    rhsStride.set(rhs.stride);
    resultStride.set(result.stride);
    innerDim[0] = lhs.shape[1];

    const metaDataUniform = device!.createBuffer({
      label: 'Matmul Meta Data',
      size: (4 * 2) * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device!.queue.writeBuffer(metaDataUniform, 0, metaDataBuffer);

    const bindGroup = device!.createBindGroup({
      label: 'Matmul Bind Group',
      layout: Matrix.matmul!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: lhs.buffer },
        { binding: 1, resource: rhs.buffer },
        { binding: 2, resource: result.buffer },
        { binding: 3, resource: metaDataUniform },
      ],
    });

    return bindGroup;
  }

  mult(other: Matrix) {
    if (this.shape[1] !== other.shape[0]) {
      throw new TypeError(`Cannot multiply matrix of shape ${this.shape} with matrix of shape ${other.shape}`);
    }
    AssertInPass();

    const result = new Matrix([this.shape[0], other.shape[1]], this.requires_grad || other.requires_grad);

    pass!.setBindGroup(0, Matrix.createMultBindGroup(this, other, result));
    pass!.setPipeline(Matrix.matmul!);
    pass!.dispatchWorkgroups(result.shape[0], result.shape[1]);

    if (result.requires_grad) {
      result.backwards = () => {
        const resultGrads = result.grads();
        if (this.requires_grad) {
          pass!.setBindGroup(0, Matrix.createMultBindGroup(resultGrads, other.transpose(), this.grads()));
          pass!.setPipeline(Matrix.matmul!);
          pass!.dispatchWorkgroups(this.shape[0], this.shape[1]);
          this.backwards();
        }
        if (other.requires_grad) {
          pass!.setBindGroup(0, Matrix.createMultBindGroup(this.transpose(), resultGrads, other.grads()));
          pass!.setPipeline(Matrix.matmul!);
          pass!.dispatchWorkgroups(other.shape[0], other.shape[1]);
          other.backwards();
        }
      }
    }

    return result;
    // TODO: Setup backward pass ^^
  }

  async loadFromGPU() {
    AssertNotInPass();

    const outputBuffer = this.readBuffer ?? device!.createBuffer({
      label: 'Load From GPU Buffer',
      size: this.buffer.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    if (!this.readBuffer) this.readBuffer = outputBuffer;

    encoder!.copyBufferToBuffer(this.buffer, outputBuffer);

    // TODO: Somehow add a check that this isn't awaited too early (maybe custom thenable?)
    await submitted;

    await outputBuffer.mapAsync(GPUMapMode.READ);
    const mappedResult = new Float32Array(outputBuffer.getMappedRange());

    const result = new Float32Array(mappedResult);

    outputBuffer.unmap();

    return result;
  }

  static fromArray(values: number[][], requires_grad: boolean = true) {
    const rows = values.length;
    const cols = values[0].length;

    const mat = new Matrix([rows, cols], requires_grad);

    const floatBuffer = new Float32Array(rows * cols);

    for (let row = 0; row < rows; ++row) {
      for (let col = 0; col < cols; ++col) {
        floatBuffer[row * cols + col] = values[row][col];
      }
    }

    device!.queue.writeBuffer(mat.buffer, 0, floatBuffer);

    return mat;
  }

  setGrads(values: number[][]) {
    const rows = values.length;
    const cols = values[0].length;
    if (!this.requires_grad) throw new TypeError('Can\'t set grads of matrix that doesn\'t requier_grad');
    if (rows !== this.shape[0] || cols !== this.shape[1]) throw new TypeError('Row and/or column mismatch when setting grads');

    const floatBuffer = new Float32Array(rows * cols);

    for (let row = 0; row < rows; ++row) {
      for (let col = 0; col < cols; ++col) {
        floatBuffer[row * cols + col] = values[row][col];
      }
    }

    device!.queue.writeBuffer(this.gradBuffer!, 0, floatBuffer);
  }

  private static matmul: GPUComputePipeline | undefined = undefined;

  static async Init(): Promise<boolean> {
    const code = await fetch('/wgsl/compiled-matmul.wgsl').then(res => res.text());
    const matmulModule = device!.createShaderModule({
      label: 'Matrix Multiplication Module',
      code,
    });

    const matmulPipeline = await device!.createComputePipelineAsync({
      label: 'Matrix Multiplication Pipeline',
      layout: 'auto',
      compute: {
        module: matmulModule,
      },
    });

    Matrix.matmul = matmulPipeline;
    return true;
  }

}
