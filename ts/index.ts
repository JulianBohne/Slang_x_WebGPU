import { Slider } from './gui.js';
import { Init, BeginPass, EndPass, Submit, Matrix, ZeroGrad } from './matrix.js';
import { Splatter } from './renderer.js';

(async () => {
  const canvas = document.getElementById('surface') as HTMLCanvasElement;

  canvas.onclick = () => {
    canvas.requestPointerLock();
  };


  const gui_container = document.getElementById('gui-container') as HTMLDivElement;

  const splatter = await Splatter.new(canvas);
  
  const degreesPerPixel = 1;
  canvas.onmousemove = e => {
    if (document.pointerLockElement == canvas) {
      const deltaYaw = e.movementX * degreesPerPixel * (Math.PI / 180);
      const deltaPitch = e.movementY * degreesPerPixel * (Math.PI / 180);
      splatter.camera.yaw += deltaYaw;
      splatter.camera.pitch += deltaPitch;
    }
  };

  new Slider(gui_container, splatter.camera, 'fov', 0.1, 0.9*Math.PI, () => {});
  new Slider(gui_container, splatter.camera, 'far', 1, 100, () => {});
  new Slider(gui_container, splatter.camera, 'near', 0.1, 10, () => {});
  new Slider(gui_container, splatter.camera, 'x', -10, 10, () => {});
  new Slider(gui_container, splatter.camera, 'y', -10, 10, () => {});
  new Slider(gui_container, splatter.camera, 'z', -10, 10, () => {});

  const animationFunc = async (time: DOMHighResTimeStamp) => {
    splatter.render(time);
    // const buf = splatter.device.createBuffer({
    //   size: splatter.splat_buffer_2D.size,
    //   usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    // });

    // const enc = splatter.device.createCommandEncoder();
    // enc.copyBufferToBuffer(splatter.splat_buffer_2D, buf);
    // const com = enc.finish();
    // splatter.device.queue.submit([com]);

    // await buf.mapAsync(GPUMapMode.READ);
    // const cpuBuf = new Float32Array(buf.getMappedRange());
    // const gaussian2D_float_size = (2 + 1 + 1 /* padding */ + 2*2 + 3 + 1 + 2 + 2);
    // const depths = cpuBuf.filter((_, i) => (i % gaussian2D_float_size) == 2);
    // const minDepth = depths.reduce((p, c) => Math.min(p, c), Number.POSITIVE_INFINITY);
    // const maxDepth = depths.reduce((p, c) => Math.max(p, c), Number.NEGATIVE_INFINITY);
    // const avgDepth = depths.reduce((p, c) => p + c, 0) / depths.length;
    // console.log(minDepth, avgDepth, maxDepth);

    requestAnimationFrame(animationFunc);
  };
  requestAnimationFrame(animationFunc);

})();

(async () => {
  const canvas = document.getElementById('surface') as HTMLCanvasElement;

  if (!await Init(canvas)) {
    console.error('GPU initialization failed');
    return;
  }

  const a = Matrix.fromArray([
    [1, 2, 3],
    [4, 5, 6],
  ]);

  const b = Matrix.fromArray([[7, 8, 9]]).transpose();

  BeginPass();

  const result = a.mult(b);

  result.backwards();

  ZeroGrad();

  result.backwards();
  result.backwards();

  EndPass();

  const aGrad = a.grads().loadFromGPU();

  Submit();


  console.log(await aGrad);

});
