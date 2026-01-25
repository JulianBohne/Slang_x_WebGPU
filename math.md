# Projection Marix ^^

## Definitions:
$\newcommand{\mat}{\mathbf}$
$\renewcommand{\vec}{\mathbf}$
- $\theta$ is the horizontal field of view
- $a$ is the aspect ratio (e.g. 16/9)
- $n$ is the near plane distance
- $f$ is the far plane distance

## Requirements:
- The point $(0, 0, n)$ should be mapped to $(0, 0, 0)$
- The point $(0, 0, f)$ should be mapped to $(0, 0, 1)$
- A point $(a \cdot t \cdot \tan \theta/2, 0, t)$ should be mapped to some $(1, 0, ?)$
- The operation should look like $\frac{\mat P \vec x}{|(\mat P \vec x)_w|}$

## Deriving the columns
$$
\begin{bmatrix}
  b & 0 & 0 & 0 \\
  0 & h & 0 & 0 \\
  0 & 0 & m & o \\
  0 & 0 & 1 & 0
\end{bmatrix}
$$

$$
\begin{align*}
\frac{m \cdot n + o}{n} \overset{!}{=} 0 \\
m \cdot n + o = 0 \\
o = -m \cdot n \\
o = -\frac{f \cdot n}{f - n} \\
\end{align*}
$$

$$
\begin{align*}
\frac{m \cdot f + o}{f} \overset{!}{=} 1 \\
m \cdot f + o = f \\
m \cdot f - m \cdot n = f \\
m \cdot (f - n) = f \\
m = \frac{f}{f - n} \\
\end{align*}
$$

$$
\begin{align*}
\frac{b \cdot a \cdot t \cdot \tan \theta/2}{t} \overset{!}{=} 1 \\
b \cdot a \cdot \tan \theta/2 = 1 \\
b = \frac{1}{a \cdot \tan \theta/2} \\
\end{align*}
$$

$$
\mat P =
\begin{bmatrix}
  \frac{1}{a \cdot \tan \theta/2} & 0 & 0 & 0 \\
  0 & \frac{1}{\tan \theta/2} & 0 & 0 \\
  0 & 0 & \frac{f}{f - n} & -\frac{f \cdot n}{f - n} \\
  0 & 0 & 1 & 0
\end{bmatrix}
$$

## Jacobian at some point $\vec p = (x, y, z)$

$\vec p' = (\frac{x \cdot b}{z}, \frac{y \cdot h}{z}, \frac{z \cdot m + o}{z}) = (\frac{x \cdot b}{z}, \frac{y \cdot h}{z}, m + \frac{o}{z})$

$$\frac{\partial \vec p'}{\partial x} = (\frac{b}{z}, 0, 0)$$
$$\frac{\partial \vec p'}{\partial y} = (0,\frac{h}{z}, 0)$$
$$\frac{\partial \vec p'}{\partial z} = (-\frac{x \cdot b}{z^2}, -\frac{y \cdot h}{z^2}, -\frac{o}{z^2})$$

$$
\mat J =
\begin{bmatrix}
\frac{b}{z} & 0 & - \frac{x \cdot b}{z^2} \\
0 & \frac{h}{z} & - \frac{y \cdot h}{z^2} \\
0 & 0 & - \frac{o}{z^2} \\
\end{bmatrix}
$$

$$
\begin{bmatrix}
  b & 0 & 0 & 0 \\
  0 & h & 0 & 0 \\
  0 & 0 & m & o \\
  0 & 0 & 1 & 0
\end{bmatrix}
$$
