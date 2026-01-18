
var device: GPUDevice | undefined = undefined;
var encoder: GPUCommandEncoder | undefined = undefined;
var pass: GPUComputePassEncoder | undefined = undefined;

var resolveSubmit: () => void = () => {};
var submitted: Promise<void> | undefined = undefined;

var gradientMatrices: Matrix[] = [];

export async function Init(): Promise<boolean> {
  if (!('gpu' in navigator)) return false;

  const adapter = await navigator.gpu.requestAdapter({
    featureLevel: 'core'
  });
  if (!adapter) return false;

  if (adapter.limits.maxBindGroups < 3) {
    console.error(`A minimum of 3 bind groups is required for this application to work, but this GPU only supports ${adapter.limits.maxBindGroups}`);
    return false;
  }

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

export function ZeroGrad() {
  // TODO: Do this with a kernel
  for (const gradMat of gradientMatrices) {
    console.log('Zeroing', gradMat);
    gradMat.zero();
  }
}

export class Matrix {

  shape: [number, number];
  stride: [number, number];

  buffer: GPUBuffer;
  gradMat: Matrix | undefined;
  readBuffer: GPUBuffer | undefined;

  roBindGroup: GPUBindGroup;
  rwBindGroup: GPUBindGroup;

  requires_grad: boolean;

  backwardsFunction: () => void;

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

      this.backwardsFunction = () => {};

      this.readBuffer = undefined;

      this.generateBindGroups(); // This sets the ro and rwBindGroup, so hacking a bit lol
      this.roBindGroup = (this as Matrix).roBindGroup;
      this.rwBindGroup = (this as Matrix).rwBindGroup;

      if (this.requires_grad) {
        const gradMat = new Matrix(this);
        gradMat.requires_grad = false;
        gradMat.buffer = device!.createBuffer({
          label: 'Gradient Buffer',
          size: 4 * rows * cols,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        gradMat.generateBindGroups();

        this.gradMat = gradMat;

        gradientMatrices.push(gradMat);
      }
    } else {
      const other = arg;
      this.shape = other.shape;
      this.stride = other.stride;
      this.buffer = other.buffer;
      this.readBuffer = other.readBuffer;
      this.requires_grad = other.requires_grad;
      this.backwardsFunction = other.backwardsFunction;
      this.roBindGroup = other.roBindGroup;
      this.rwBindGroup = other.rwBindGroup;
      this.gradMat = other.gradMat;
    }
  }

  generateBindGroups() {
    // https://webgpufundamentals.org/webgpu/lessons/resources/wgsl-offset-computer.html

    const metaDataBuffer = new Uint32Array([...this.shape, ...this.stride]);

    const metaDataUniform = device!.createBuffer({
      label: 'Matmul Meta Data',
      size: metaDataBuffer.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device!.queue.writeBuffer(metaDataUniform, 0, metaDataBuffer);

    this.roBindGroup = device!.createBindGroup({
      label: 'Matrix Read Only Bind Group',
      layout: Matrix.roBindGroupLayout!,
      entries: [
        { binding: 0, resource: metaDataUniform },
        { binding: 1, resource: this.buffer },
      ],
    });

    this.rwBindGroup = device!.createBindGroup({
      label: 'Matrix Read / Write Bind Group',
      layout: Matrix.rwBindGroupLayout!,
      entries: [
        { binding: 0, resource: metaDataUniform },
        { binding: 1, resource: this.buffer },
      ],
    });
  }

  backwards() {
    AssertInPass();
    if (!this.requires_grad) throw new TypeError('Can\'t call backward on a matrix without requires_grad');

    pass!.setBindGroup(0, this.gradMat!.rwBindGroup);
    pass!.setPipeline(Matrix.onePipeline!);
    pass!.dispatchWorkgroups(this.gradMat!.shape[0], this.gradMat!.shape[1]);

    // TODO: Do this with a kernel?
    const floatBuffer = new Float32Array(this.shape[0] * this.shape[1]).fill(1);
    device!.queue.writeBuffer(this.gradMat!.buffer!, 0, floatBuffer);

    this.backwardsFunction();
  }

  transpose() {
    const result = new Matrix(this);
    result.shape = [this.shape[1], this.shape[0]];
    result.stride = [this.stride[1], this.stride[0]];
    result.generateBindGroups();
    if (result.gradMat) { // TODO: Figure something out so I don't have to do something error prone like this ^^'
      result.gradMat = new Matrix(result.gradMat);
      result.gradMat.shape = [ result.gradMat.shape[1], result.gradMat.shape[0] ];
      result.gradMat.stride = [ result.gradMat.stride[1], result.gradMat.stride[0] ];
      result.gradMat.generateBindGroups();
    }
    return result;
  }

  grads() {
    if (!this.requires_grad) throw new TypeError('Can\'t convert matrix that doesn\'t requiers_grad to gradient target');
    return this.gradMat!;
  }

  mult(other: Matrix) {
    if (this.shape[1] !== other.shape[0]) {
      throw new TypeError(`Cannot multiply matrix of shape ${this.shape} with matrix of shape ${other.shape}`);
    }
    AssertInPass();

    const result = new Matrix([this.shape[0], other.shape[1]], this.requires_grad || other.requires_grad);

    pass!.setBindGroup(0, result.rwBindGroup);
    pass!.setBindGroup(1, this.roBindGroup);
    pass!.setBindGroup(2, other.roBindGroup);
    pass!.setPipeline(Matrix.forwardsPipeline!);
    pass!.dispatchWorkgroups(result.shape[0], result.shape[1]);

    if (result.requires_grad) {
      result.backwardsFunction = () => {
        const resultGrads = result.grads();
        if (this.requires_grad) {
          pass!.setBindGroup(0, this.grads().rwBindGroup);
          pass!.setBindGroup(1, resultGrads.roBindGroup);
          pass!.setBindGroup(2, other.transpose().roBindGroup);

          pass!.setPipeline(Matrix.backwardsPipeline!);
          pass!.dispatchWorkgroups(this.shape[0], this.shape[1]);
          this.backwardsFunction();
        }
        if (other.requires_grad) {
          pass!.setBindGroup(0, other.grads().rwBindGroup);
          pass!.setBindGroup(1, this.transpose().roBindGroup);
          pass!.setBindGroup(2, resultGrads.roBindGroup);

          pass!.setPipeline(Matrix.backwardsPipeline!);
          pass!.dispatchWorkgroups(other.shape[0], other.shape[1]);
          other.backwardsFunction();
        }
      }
    }

    return result;
    // TODO: Setup backward pass ^^
  }

  zero() {
    AssertInPass();
    pass!.setBindGroup(0, this.rwBindGroup);
    pass!.setPipeline(Matrix.zeroPipeline!);
    pass!.dispatchWorkgroups(this.shape[0], this.shape[1]);
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

  private static forwardsPipeline: GPUComputePipeline | undefined = undefined;
  private static backwardsPipeline: GPUComputePipeline | undefined = undefined;
  private static zeroPipeline: GPUComputePipeline | undefined = undefined;
  private static onePipeline: GPUComputePipeline | undefined = undefined;
  private static roBindGroupLayout: GPUBindGroupLayout | undefined = undefined;
  private static rwBindGroupLayout: GPUBindGroupLayout | undefined = undefined;

  static async Init(): Promise<boolean> {
    const code = await fetch('/wgsl/compiled-matrix.wgsl').then(res => res.text());
    const matrixModule = device!.createShaderModule({
      label: 'Matrix Multiplication Module',
      code,
    });


    const roBindGroupLayout = device!.createBindGroupLayout({
      label: '(RO)Matrix Bind Group Layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'uniform', // Matrix meta data uniform
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'read-only-storage', // Matrix buffer to read (only)
          },
        },
      ],
    });
    Matrix.roBindGroupLayout = roBindGroupLayout;

    const rwBindGroupLayout = device!.createBindGroupLayout({
      label: 'RWMatrix Bind Group Layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'uniform', // Matrix meta data uniform
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage', // Matrix buffer to read / write
          },
        },
      ],
    });
    Matrix.rwBindGroupLayout = rwBindGroupLayout;

    const binOpPipelineLayout = device!.createPipelineLayout({
      label: 'Binary Operator Matrix Pipeline Layout',
      bindGroupLayouts: [
        rwBindGroupLayout, // Result
        roBindGroupLayout, // LHS
        roBindGroupLayout, // RHS
      ],
    });

    const forwardsPipeline = await device!.createComputePipelineAsync({
      label: 'Matrix Multiplication Forwards Pipeline',
      layout: binOpPipelineLayout,
      compute: {
        module: matrixModule,
        entryPoint: 'matmul_forwards',
      },
    });
    Matrix.forwardsPipeline = forwardsPipeline;

    const backwardsPipeline = await device!.createComputePipelineAsync({
      label: 'Matrix Multiplication Backwards Pipeline',
      layout: binOpPipelineLayout,
      compute: {
        module: matrixModule,
        entryPoint: 'matmul_backwards',
      },
    });
    Matrix.backwardsPipeline = backwardsPipeline;

    const selfOpPipelineLayout = device!.createPipelineLayout({
      label: 'Self Operation Matrix Pipeline Layout',
      bindGroupLayouts: [
        rwBindGroupLayout, // Self
      ],
    });

    const zeroPipeline = await device!.createComputePipelineAsync({
      label: 'Zero Matrix Pipeline',
      layout: selfOpPipelineLayout,
      compute: {
        module: matrixModule,
        entryPoint: 'set_zero',
      },
    });
    Matrix.zeroPipeline = zeroPipeline;

    const onePipeline = await device!.createComputePipelineAsync({
      label: 'Set Matrix to One Pipeline',
      layout: selfOpPipelineLayout,
      compute: {
        module: matrixModule,
        entryPoint: 'set_one',
      },
    });
    Matrix.onePipeline = onePipeline;

    return true;
  }

}
