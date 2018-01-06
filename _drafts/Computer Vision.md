# What is Image
- A array of data
- A function $I(x, y)$
- A 2D projection of 3D points

# Filtering
## Weighted Moving Average
the number of weights should be **odd**, makes it easier to have a middle pixel

## Correlation filtering — nonuniform weights

<p>

$$
G[i, j] = \sum_{u=-k}^{k}\sum_{v=-k}^{k}H[u,v] I[i+u, j+v]
$$

</p>
Correlation filtering
where, $H[u,v]$ is a filter, which is called **kernel** or **mask**.

## Gaussian Filter

<p>

$$
g(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
$$

</p>

The most import parameter of gaussian filter is $\sigma$, which define the scope and intensity of filtering.

## Unsharp Mask(USM)
图像的锐化处理,使图像的边缘更加清晰

## Noise
- Gaussian noise: a common noise
- salt-and-pepper noise(椒盐噪声/脉冲噪声): 图像中有很多“噪点”
    Use **median filter (中值滤波)** to filt this noise



***References***
- [wiki: 椒盐噪声](https://zh.wikipedia.org/wiki/%E6%A4%92%E7%9B%90%E5%99%AA%E5%A3%B0)


# Edge Detection
## Some well-know gradient masks

## The Laplacian

<p>

$$
h(f) = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}
$$

</p>

## Canny edge operator
1. Filter image with derivative of Gaussian
2. Find magnitude and orientation of gradient
3. Non-maximum suppression:
    Thin multi-pixel wide "ridges" down to single pixel width
4. Linking and thresholding
    - Define two thresholds: low and high
    - Use the high threshold to start edge curves and the low threshold to continue them

# Hough Transform
## Hough Space
The Hough space is **(a parameters space)**, which use model(line, circle) parameters as axis, such as $m, b$ in $y = mx + b$ or $r, d$ in $x\cos\theta + y\sin\theta = r$.

**Why use $x\cos\theta + y\sin\theta = r$**: When using slope-intercept form, if a line is vertical, it is hard to represent it in hough space, so people use polar-coordinates-parameters line model to express line and convert it into hough space.

> A line in a image responds to a point in Hough space, a point in a image responds to a line/curve in Hough space.

## Hough Transform
**Key idea:** Find shape(line, circle or other non-analytic models) from points using Hough Transform. Hough Transform transforms each point in image into hough space and slice hough space axis into bins, and using **vote** to find which parameters can determine a shape.

<p>

$$
r = x\cos\theta + y\sin\theta
$$

</p>

<p align="center">
    <img src="http://ovvybawkj.bkt.clouddn.com/cv/hough-space-polar-coordinates.png" width="40%">
</p>

For this simple Hough Transform, when you want to find lines in a image with Hough transform, first you should generate edge image(for example using Canny), and then use Hough transform.

**Disadvantage:**
- Complexity of search time increases exponentially with the number of model parameters
- Non-target shapes can produce spurious peaks in parameter space
- Quantization: hard to pick a good grid size

***References***
- [霍夫变换 ( Hough Transform）直线检测](http://www.cnblogs.com/Ponys/p/3146753.html)
- [OpenCV doc: Hough Line Transform](https://docs.opencv.org/3.3.1/d6/d10/tutorial_py_houghlines.html)

## Generalized Hough Transform
Use "visual codeword" (features from image) as a "edge point" in original Hough Transform, and calculate the displacement between those "visual code words" and center-point of the object, and then to find the shape you want.
- Training
    1.use cluster algorithem to generate features(these features can be regarded as a oprator, and when use them on image, they will generate **interest points**). And then get thier displacement of the center-point of the object.


# Fourier Transform
# Fourier Series
Any periodic function can be rewritten as a weighted sum of sines and cosines of different frequencies.

## Fourier Transform
This transform reparametrizes the signal by $\omega$ instead of $x$， **spatial Domain** to **Frequency Domain**:

> When we do a Fourier transform, all we're doing is computing a basis set.

The infinite intergral of the product of two sinusoids of different frequency is zero
<p>

$$
\int_{-\infty}^{\infty}\sin(ax + \phi)\sin(bx + \psi)dx = 0, ~\mathrm{if} ~ a \ne b
$$

</p>

And if same feequency the integral is infinite:
<p>

$$
\int_{-\infty}^{\infty}sin(ax + \phi)sin(ax+\psi)dx = \pm\infty,
$$
if $\phi$ and $\psi$ not exactly $\frac{\pi}{2}$ out of phase($\sin$ and $\cos$)

</p>

<p>

$$
F(\omega) = \int_{-\infty}^{\infty}f(x)e^{-i2\pi\omega x}dx 
$$
where, $e^{ik} = \cos k + i\sin k$, $i=\sqrt{-1}$

</p>

**理解：** 通过计算乘积的积分来判断哪些频率点是有值的（积分不为0），输出的结果是信号在每个频段的响应。换个角度理解就是找到频域中的基去表达输入的信号。

## Discrete Fourier Transform
<p>

$$
F(k) = \frac{1}{N}\sum_{x=0}^{x=N-1}f(x)e^{-i\frac{2\pi kx}{N}}
$$

</p>

> Sampling is just multiplying the continuous signal by the discrete comb.

> The less often we sample in space, the higher the samples in frequency.

# Camera
- aperture: change depth
- lens: change field of view, how wide of a view do we have

<p>

$$
\frac{1}{z^{'}} + \frac{1}{z} = \frac{1}{f}
$$

</p>

Any object point satisfying this equation is in focus. So by moving the lens in and our a little bit from the CCD, we can change where in the world things are in focus.

Some problem
- vignetting(晕影)
- chromatic aberration(色差)
- geometric distortion(几何变形)

## Pinhole model(针孔模型)
### Modeling Projection - Coordinate System
<p align="center">
    <img src="http://ovvybawkj.bkt.clouddn.com/cv/model_projection-coordinate_system.png" width="40%">
</p>

COP: Center of projection, it is optical center
PP: Projection plane or image plane

In the mathematical model, we put the image palne in front of the COP, so the object will not be inverted in image and the $y$ need not to be reciprocal.

<p>

$$
(X, Y, Z) \rightarrow (-d\frac{X}{Z},-d\frac{Y}{Z}, -d)
$$

where (X,Y,Z) are in camera coordinates, (x, y) are in image coordinates.
</p>

## Homogeneous Coordinates
***References***
- [3D数学基础-矩阵变换（二）](http://frankorz.com/2017/09/24/matrix-transformation-2/)

we put image plane at (0, 0, f)

## Perspective Projection
f is focal length: put image plane at (0, 0, f)

### Vanishing points
Sets of parallel lines on the same plane lead to collinear(共线的) vanishing points. This line is called the horizon for that plane.
<p align="center">
    <img src="http://ovvybawkj.bkt.clouddn.com/cv/horizon_of_plane.png" width="50%">
</p>

For a image example,
<p align="center">
    <img src="http://ovvybawkj.bkt.clouddn.com/cv/horizon_of_plane_example.png" width="50%">
</p>

### Other models:
One model 对应一个 projection matrix

- Orthograpphic projection
Parallel projection: special case of perspective projection

- Weak Perspective
Assume all object in the world has the same depth(depth over an object or over the range of an object is very samll compared to the difference in depth from the object to the center of projection)

# Stereo geometry
## Estimating depth with stereo
Two need have:
- Calibrated cameras
- Matched points

## Epipolar geometry
**理解** epipolar line的理解，一切基于针孔模型，两张图像（image plane），相机坐标系中的3D的某一个点在左图的投影的连线，与其在右图投影所在点的连线，构成了平面，该平面与两个image plane的交线就是epipolar line


## Stereo correspondence
Using epipolar line, search region is made to samller as one demension

Porblem:
- Occlusion: half occlusion, one camera can see but another can not see

Energy minimization problem?

Regard stereo depth calculate(stereo disparity problem) as an energy minimization problem and the energy function of this form can be minimized using **graph cuts**


> Fast Approximate energy minimization via graph cuts, PAMI 2001

Challenges:
- Low-contrast
- Occlusions
- Violations of brightness constancy(e.g. specular reflections)
- Really large baselines(two camera far away each other)(foreshortening and apperance change)
- Camera calibration errors(it will make epipolar lines error)

# Camera
- [Udacity - Computer vision: Extrinsic camera parameters](https://classroom.udacity.com/courses/ud810/lessons/2952438768/concepts/29548388600923)
    A summry of camera model, homogeneous coordinates and perspective projection.
## Extrinsic camera parameters
### Geometric camera calibration
We need the relationship between coordinates in the world and coordinates in the image: **geometric camera calibration**, it is composed of 2 transformations:
- **Extrinsic parameters(or camera pose):** From world coordinate system to the camera's 3D coordinate system
- **Intrinsic parameters:** From the 3D coordinates in the camera frame to the 2D image plane via projection