
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
    ) {
        const sliderContainer = document.createElement('div');
        sliderContainer.style.display = 'flex';
        sliderContainer.style.flexDirection = 'column';
        sliderContainer.style.alignContent = 'center';

        const label = sliderContainer.appendChild(document.createElement('label'));
        label.textContent = String(property);

        const sliderInput = sliderContainer.appendChild(document.createElement('input'));
        sliderInput.type = 'range';
        sliderInput.min = String(min);
        sliderInput.max = String(max);
        sliderInput.step = '0.1'; // Make customizable?
        const thing = object[property];
        sliderInput.value = String(thing);


        sliderInput.addEventListener('input', () => {
            (object[property] as unknown as number) = Number(sliderInput.value);
            callback();
        });
        container.appendChild(sliderContainer);
    }
}
