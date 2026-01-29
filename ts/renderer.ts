
const float_size = 4;
const mat4_size = 4*4 * float_size;

type MatrixValues = [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
];

class Matrix {
  values: MatrixValues; // TODO: Transpose this shait

  constructor(values: MatrixValues) {
    this.values = values;
  }

  static Identity(): Matrix {
    return new Matrix([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ]);
  }

  // static RotationX(angle: number): Matrix {

  //   return new Matrix([
  //     1, 0, 0, 0,
  //     0, 
  //   ]);
  // }

}

class SplatCamera {

  static size = mat4_size /* Model View */ + mat4_size /* Projection */;

  buffer: GPUBuffer;

  device: GPUDevice;

  fov: number;
  aspect_ratio: number;
  far: number;
  near: number;

  pitch: number;
  yaw: number;

  x: number;
  y: number;
  z: number;

  constructor(device: GPUDevice) {
    this.buffer = device.createBuffer({
      label: 'SplatCamera Constant Buffer',
      size: SplatCamera.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM,
    });

    this.device = device;

    this.fov = Math.PI / 2; // 90 degrees
    this.aspect_ratio = 1; // 1 : 1
    this.far = 1000;
    this.near = 1;

    this.pitch = 0;
    this.yaw = 0;

    this.x =  0;
    this.y =  0;
    this.z = -5;

    this.to_gpu();
  }

  to_gpu() {
    const cb = new Float32Array(new ArrayBuffer(SplatCamera.size));

    // Model View Matrix (Note the matrix has to be written transposed here)
    cb.set([
      1, 0,  0, 0,
      0, 1,  0, 0,
      0, 0,  1, 0,
      -this.x, -this.y, -this.z, 1,
    ]);
    
    const tf = Math.tan(this.fov / 2);

    const a = this.aspect_ratio;
    const f = this.far;
    const n = this.near;

    // Projection Looking in +z -> NDC [-1,1][-1,1][0,1] (Note the matrix has to be written transposed here)
    cb.set([
      1 / (a * tf),      0,                  0, 0,
                 0, 1 / tf,                  0, 0,
                 0,      0,        f / (f - n), 1,
                 0,      0, -(f * n) / (f - n), 0,
    ], 4*4);
    
    this.device.queue.writeBuffer(this.buffer, 0, cb); // Write camera to GPU
  }
}

function angleAxisToQuat([x, y, z]: [number, number, number]) {
  const angle = Math.sqrt(x * x + y * y + z * z);
  const s = Math.sin(angle);
  const c = Math.cos(angle);
  return [s*x/angle, s*y/angle, s*z/angle, c];
}

class Gaussian3D {
  // float3 position;
  static position_offset = 0;

  // float4 rotation;
  static rotation_offset = Gaussian3D.position_offset + 4 * 4;

  // float3 scale;
  static scale_offset = Gaussian3D.rotation_offset + 4 * 4;

  // float3 color;
  static color_offset = Gaussian3D.scale_offset + 4 * 4;

  // float alpha;
  static alpha_offset = Gaussian3D.color_offset + 3 * 4;

  static size = Gaussian3D.alpha_offset + 1 * 4;
  
  constructor(cpu_buffer: ArrayBuffer, index: number) {

    const offset = index * Gaussian3D.size;

    const position_buffer = new Float32Array(cpu_buffer, offset + Gaussian3D.position_offset, 3);
    const rotation_buffer = new Float32Array(cpu_buffer, offset + Gaussian3D.rotation_offset, 4);
    const scale_buffer    = new Float32Array(cpu_buffer, offset + Gaussian3D.scale_offset,    3);
    const color_buffer    = new Float32Array(cpu_buffer, offset + Gaussian3D.color_offset,    3);
    const alpha_buffer    = new Float32Array(cpu_buffer, offset + Gaussian3D.alpha_offset,    1);

    const spread = 1;

    position_buffer.set([(Math.random() - 0.5) * spread, (Math.random() - 0.5) * spread, (Math.random() - 0.5) * spread]);
    rotation_buffer.set(angleAxisToQuat([Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1]));
    scale_buffer.set([Math.random()*1, Math.random()*1, Math.random()*1]);
    color_buffer.set([Math.random(), Math.random(), Math.random()]);
    alpha_buffer.set([1]); // Full opacity
  }
}

let counter = 0;

export class Splatter {

  static async new(target_canvas: HTMLCanvasElement): Promise<Splatter> {
    if (!('gpu' in navigator)) throw new Error('WebGPU not supported');

    const vertex_shader_source_promise = fetch('/wgsl/compiled-splat-vertex.wgsl').then(res => res.text());
    const fragment_shader_source_promise = fetch('/wgsl/compiled-splat-fragment.wgsl').then(res => res.text());
    const render_shader_source_promise = fetch('/wgsl/compiled-splat-render.wgsl').then(res => res.text());

    const adapter = await navigator.gpu.requestAdapter({ featureLevel: 'core' });
    if (!adapter) throw new Error('WebGPU not supported');

    const device = await adapter!.requestDevice();

    const shader_sources = await Promise.all([vertex_shader_source_promise, fragment_shader_source_promise, render_shader_source_promise]);

    return new Splatter(target_canvas, device, shader_sources)
  }

  render(current_time: DOMHighResTimeStamp) {
    this.camera.to_gpu();

    const compute_encoder = this.device.createCommandEncoder({ label: 'Splatting Compute Encoder' });
    const compute_pass = compute_encoder.beginComputePass({ label: 'Compute Pass' });
    compute_pass.setPipeline(this.render_pipeline);
    compute_pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.render_pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.camera,
        },
        {
          binding: 1,
          resource: this.splat_buffer_3D,
        },
        {
          binding: 2,
          resource: this.splat_buffer_2D,
        },
        {
          binding: 3,
          resource: this.render_texture,
        },
      ],
    }));
    compute_pass.dispatchWorkgroups(Math.floor((800 - 1) / 16) + 1, Math.floor((600 - 1) / 16) + 1, 1);
    compute_pass.end();
    const compute_command_buffer = compute_encoder.finish();
    this.device.queue.submit([compute_command_buffer]);

    const present_encoder = this.device.createCommandEncoder({ label: 'Splatting Presentation Encoder' });
    const pass = present_encoder.beginRenderPass(this.next_render_pass_descriptor());
    if (this.start_time === undefined) this.start_time = current_time;

    this.device.queue.writeBuffer(this.uniform_buffer, 0, new Float32Array([this.canvas.width / this.canvas.height, (current_time - this.start_time) / 1000]));

    pass.setPipeline(this.presentation_pipeline);

    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.presentation_pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.render_texture,
        },
        // {
        //   binding: 1,
        //   resource: this.sampler,
        // },
      ],
    }));
    pass.draw(6); // Full screen quad
    pass.end();

    const command_buffer = present_encoder.finish();
    this.device.queue.submit([command_buffer]);
  }

  canvas: HTMLCanvasElement;
  device: GPUDevice;
  render_pipeline: GPUComputePipeline;
  presentation_pipeline: GPURenderPipeline;
  camera: SplatCamera;
  splat_buffer_3D: GPUBuffer;
  splat_buffer_2D: GPUBuffer;
  render_texture: GPUTexture;
  // sampler: GPUSampler;
  next_render_pass_descriptor: () => GPURenderPassDescriptor;
  start_time: number | undefined;
  uniform_buffer: GPUBuffer;

  private constructor(
    target_canvas: HTMLCanvasElement,
    device: GPUDevice,
    [vertex_shader_source, fragment_shader_source, render_shader_source]: [string, string, string],
  ) {
    this.canvas = target_canvas;
    this.device = device;

    const context = target_canvas.getContext('webgpu');
    if (!context) throw new Error('Could not get webgpu context of canvas');

    const presentation_format = navigator.gpu.getPreferredCanvasFormat();

    context.configure({
      device,
      format: presentation_format,
    });

    const module_vertex = device.createShaderModule({
      label: 'Vertex Splatting Presentation Module',
      code: vertex_shader_source,
    });

    const module_fragment = device.createShaderModule({
      label: 'Fragment Splatting Presentation Module',
      code: fragment_shader_source,
    });

    const module_render = device.createShaderModule({
      label: 'Compute Splatting Module',
      code: render_shader_source,
    });

    this.render_pipeline = device.createComputePipeline({
      label: 'Compute Splatting Pipeline',
      layout: 'auto',
      compute: {
        module: module_render,
      },
    });

    this.presentation_pipeline = device.createRenderPipeline({
      label: 'Splatting Presentation Pipeline',
      layout: 'auto',
      vertex: {
        module: module_vertex,
      },
      fragment: {
        module: module_fragment,
        targets: [{ format: presentation_format }],
      }
    });

    this.camera = new SplatCamera(device);

    const num_gaussians = 1;

    const cpu_splat_buffer = new ArrayBuffer(Gaussian3D.size * num_gaussians);
    for (let i = 0; i < num_gaussians; ++i) {
      new Gaussian3D(cpu_splat_buffer, i);
    }

    this.splat_buffer_3D = device.createBuffer({
      label: '3D Splat Buffer',
      size: Gaussian3D.size * num_gaussians,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    device.queue.writeBuffer(this.splat_buffer_3D, 0, cpu_splat_buffer);

    // public float2 position;
    // public no_diff float depth;
    // public float2x2 inv_cov;

    // public float3 color;
    // public float alpha;

    // public no_diff float2 axis_a;
    // public no_diff float2 axis_b;
    const gaussian2D_size = 4 * (2 + 1 + 1 /* padding */ + 2*2 + 3 + 1 + 2 + 2);

    this.splat_buffer_2D = device.createBuffer({
      label: '2D Transformed Splat Buffer',
      size: gaussian2D_size * num_gaussians,
      usage: GPUBufferUsage.STORAGE,
    });

    this.render_texture = device.createTexture({
      format: 'rgba32float',
      size: [ 800, 600 ],
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING, // We will render into this texture and then display it in the fragment shader :)
    });

    // this.sampler = device.createSampler({
    //   label: 'Sampler',
    //   // magFilter: 'nearest',
    //   // minFilter: 'nearest',
    //   // mipmapFilter: 'nearest',
    // });

    const render_pass_descriptor = {
      label: 'Splatting Presentation Render Pass Descriptor',
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: [0., 0., 0., 1.0],
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    this.next_render_pass_descriptor = () => {
      render_pass_descriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
      return render_pass_descriptor as GPURenderPassDescriptor;
    };

    this.uniform_buffer = device.createBuffer({
      label: 'Uniform buffer',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    target_canvas.width = target_canvas.clientWidth;
    target_canvas.height = target_canvas.clientHeight;
    window.onresize = () => {
      target_canvas.width = target_canvas.clientWidth;
      target_canvas.height = target_canvas.clientHeight;
    }
  }
}
