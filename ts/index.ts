import { Init, BeginPass, EndPass, Submit, Matrix, ZeroGrad } from './matrix.js';
import { Splatter } from './renderer.js';

(async () => {
  const canvas = document.getElementById('surface') as HTMLCanvasElement;

  console.log('1');
  const splatter = await Splatter.new(canvas);

  const animationFunc = (time: DOMHighResTimeStamp) => {
    splatter.render(time);
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
