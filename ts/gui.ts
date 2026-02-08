
type IsEqual<T, U> = (T | U) extends (T & U) ? true : never;

type NumberKey<T, K extends keyof T> = T[K] extends number ? K : never;

export class Slider<T, K extends keyof T> {
  constructor(
      container: HTMLElement,
      object: T,
      property: NumberKey<T, K>,
      min: number,
      max: number,
      callback: () => void,
      label_text?: string,
  ) {
    const sliderContainer = document.createElement('div');
    sliderContainer.style.display = 'flex';
    sliderContainer.style.flexDirection = 'row';
    sliderContainer.style.alignContent = 'center';
    sliderContainer.style.justifyContent = 'center';
    sliderContainer.style.gap = '1em';

    if ((label_text ?? String(property)).length > 0) {
      const label = sliderContainer.appendChild(document.createElement('label'));
      label.textContent = label_text ?? String(property);
    }

    const sliderInput = sliderContainer.appendChild(document.createElement('input'));
    sliderInput.type = 'range';
    sliderInput.min = String(min);
    sliderInput.max = String(max);
    sliderInput.step = '0.01'; // Make customizable?
    const thing = object[property];
    sliderInput.value = String(thing);


    sliderInput.addEventListener('input', () => {
      console.log('hey');
      (object[property] as unknown as number) = Number(sliderInput.value);
      callback();
    });
    container.appendChild(sliderContainer);
  }
}
