import { Init, BeginPass, EndPass, Submit, Matrix, ZeroGrad } from "./matrix.js";

(async () => {
  if (!await Init()) {
    console.error('GPU initialization failed');
    return;
  }

  const a = Matrix.fromArray([
    [ 1, 2, 3 ],
    [ 4, 5, 6 ],
  ]);

  const b = Matrix.fromArray([[7, 8, 9]]).transpose();

  BeginPass();

  const result = a.mult(b);

  ZeroGrad();

  result.backwards();

  EndPass();

  const bGrad = b.grads().loadFromGPU();

  Submit();

  console.log(await bGrad);

})();
