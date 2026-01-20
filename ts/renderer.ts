
export class Splatter {

  static async new(target_canvas: HTMLCanvasElement): Promise<Splatter> {
    if (!('gpu' in navigator)) throw new Error('WebGPU not supported');

    const vertex_shader_source_promise = fetch('/wgsl/compiled-splat-vertex.wgsl').then(res => res.text());
    const fragment_shader_source_promise = fetch('/wgsl/compiled-splat-fragment.wgsl').then(res => res.text());

    const adapter = await navigator.gpu.requestAdapter({ featureLevel: 'core' });
    if (!adapter) throw new Error('WebGPU not supported');

    const device = await adapter!.requestDevice();

    const shader_sources = await Promise.all([vertex_shader_source_promise, fragment_shader_source_promise]);

    return new Splatter(target_canvas, device, shader_sources)
  }

  render(current_time: DOMHighResTimeStamp) {
    const encoder = this.device.createCommandEncoder({ label: 'Splatting Presentation Encoder' });
    const pass = encoder.beginRenderPass(this.next_render_pass_descriptor());
    if (this.start_time === undefined) this.start_time = current_time;

    this.device.queue.writeBuffer(this.uniform_buffer, 0, new Float32Array([this.canvas.width / this.canvas.height, (current_time - this.start_time) / 1000]));

    pass.setPipeline(this.pipeline);

    pass.setBindGroup(0, this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: this.uniform_buffer,
        }
      ],
    }));
    pass.draw(6);
    pass.end();

    const command_buffer = encoder.finish();
    this.device.queue.submit([command_buffer]);
  }

  canvas: HTMLCanvasElement;
  device: GPUDevice;
  pipeline: GPURenderPipeline;
  next_render_pass_descriptor: () => GPURenderPassDescriptor;
  start_time: number | undefined;
  uniform_buffer: GPUBuffer;

  private constructor(
    target_canvas: HTMLCanvasElement,
    device: GPUDevice,
    [vertex_shader_source, fragment_shader_source]: [string, string],
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
    })

    const module_fragment = device.createShaderModule({
      label: 'Fragment Splatting Presentation Module',
      code: fragment_shader_source,
    })

    this.pipeline = device.createRenderPipeline({
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
