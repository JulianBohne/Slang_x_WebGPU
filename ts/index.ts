import { Init, BeginPass, EndPass, Submit, Matrix } from "./matrix.js";

(async () => {
  if (!await Init()) {
    console.error('GPU initialization failed');
    return;
  }

  const a = Matrix.fromArray([
    [ 1, 2, 3 ],
  ]);

  const b = Matrix.fromArray([[7], [8], [9]]);

  BeginPass();

  const result = a.mult(b);
  result.setGrads([[1]]);
  result.backwards();

  EndPass();
  const bGrad = b.grads().loadFromGPU();
  Submit();
  console.log(await bGrad);

})();
