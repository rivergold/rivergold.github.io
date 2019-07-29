# Function

## Rotate Image

- `cv::getRotationMatrix2D(Poiont2f center, double angle double scale)`
- `cv::warpAffine`

### C++

```C++
Mat img_dst;
Point2f center(img_src.cols / 2.0, img_src.rows / 2.0);
Mat R = getRotationMatrix2D(center, degree, 1.0);
warpAffine(img_src, img_dst, R, img_src.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128, 128, 128));
```

### Python

```python
center = (img.shape[1] / 2, img.shape[0] / 2)
degree = 30
R = cv2.getRotationMatrix2D(center, degree, 1)
dst = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]),
                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=(128,128,128))
```

These method above will crop image as the same size of the input image. The following will show how to **do rotate without crop:**<br>

### C++

```c++
Mat img_dst;
Point2f center(img_src.cols / 2.0, img_src.rows / 2.0);
Mat R = getRotationMatrix2D(center, degree, 1.0);
// Get rotated image box
Rect box = RotateRect(center, img_src.size(), degree).boundingRect();
// These two lines code just do traonsformation
// first set rotate point as image center (- center)
// then move image center to  box center (+ box)
R.at<double>(0, 2) += box.width / 2.0 - center.x;
R.at<double>(1, 2) += box.height / 2.0 - center.y;

warpAffine(img_src, img_dst, R, box.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128, 128, 128));
```

### Python

```python
def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat
```

**_References:_**

- [OpenCV doc OpenCV-Python Tutorials: Image Processing in OpenCV: Geometric Transformations of Images](https://docs.opencv.org/3.2.0/da/d6e/tutorial_py_geometric_transformations.html)
- [OpenCV doc: warpAffine](https://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983)
- [stackoverflow: Rotate an image without cropping in OpenCV in C++](https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c)

### Rotate points on image

```python
img_h, img_w = raw_img.shape[:2]
# rotate_M shape is (2, 3)
rotate_M = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), 30, 1)
img_rotated = cv2.warpAffine(raw_img, rotate_M, (img_w, img_h))
points = np.array(landmarks_106).reshape(106, 2)
# points_ones shape is (106, 3)
points_ones = np.hstack([points, ones])
transformed_points = rotate_M.dot(points_ones.T).T
```

**_References:_**

- [stackoverflow: How can I remap a point after an image rotation?](https://stackoverflow.com/a/38794480)

## Eroding and Dilating

**_References:_**

- [OpenCV Tutorial Image Processing: Eroding and Dilating](https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html)

## HSV

**_References:_**

- [OpenCV-Python Tutorials Image Processing in OpenCV: Changing Colorspaces](https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html)
- [stackoverflow: Exact Skin color HSV range](https://stackoverflow.com/a/8757076/4636081)

## Object Tracking in OpenCV

**_References:_**

- [Learn OpenCV: Object Tracking using OpenCV (C++/Python)](https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/)

## Find Contours and max area object in image

- 首先通过 findContours 函数找到二值图像中的所有边界(这块看需要调节里面的参数)
- 然后通过 contourArea 函数计算每个边界内的面积
- 最后通过 fillConvexPoly 函数将面积最大的边界内部涂成背景

```python
b, g, r, a cv2.split(img)
kernel = np.one(30, 30), np.uint8)
a = cv2.morphology(a, cv2.MORPH_OPENkernel)

a_theshed, contour hierarchy cv2.findContours(a1, 2)
if len(contours) 1:
    c = m(contourskey=cvcontourArea)
    a_max_area np.zero(img.shape[:2] np.uint8)
    cvfillConvexPo(a_max_area, c1)
    a *= a_max_area
img = cv2.merge([bg, r, a])
```

Ref [CSDN: opencv 获取图像最大连通域 c++和 python 版](https://blog.csdn.net/xuyangcao123/article/details/81023732)

**_References:_**

- [stackoverflow: Find and draw the largest contour in opencv on a specific color (Python)](https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python)

- [OpenCV doc: OpenCV-Python Tutorials: Contour Features](https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html)

# Image file reading and writing

## `imread`

### Read image with alpha channel

```python
image = cv2.imread(<image_path>, flags=cv2.IMREAD_UNCHANGED)
```

Ref [OpenCV doc: Image file reading and writing - ImreadModes](https://docs.opencv.org/3.4.6/d4/da8/group__imgcodecs.html#ga61d9b0126a3e57d9277ac48327799c80)

**_References:_**

- [CSDN: opencv imread 读取 alpha 通道](https://blog.csdn.net/jazywoo123/article/details/17353069)

# Core Functionality

## Operations on arrays

### `addWeighted`

Compose two image with alpha channel.

Ref [OpenCV ansers: How to overlay an PNG image with alpha channel to another PNG?](https://answers.opencv.org/question/73016/how-to-overlay-an-png-image-with-alpha-channel-to-another-png/)

### `add`

Add two image.

[application] Alpha bleading

Ref [Learn OpenCV: Alpha Blending using OpenCV (C++ / Python)](https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/)

# Image Processing

## Drawing Functions

- [Doc](https://docs.opencv.org/3.4.5/d6/d6e/group__imgproc__draw.html)

**These function will change raw input image.**

**_Ref:_** [OpenCV-Python Tutorials-Gui Features in OpenCV: Drawing Functions in OpenCV](https://docs.opencv.org/master/dc/da5/tutorial_py_drawing_functions.html)

## Histograms

### `calcHist`

**_Ref:_** [CSDN: OpenCV Python 教程（3、直方图的计算与显示）](https://blog.csdn.net/sunny2038/article/details/9097989)

## Contours

- [Finding contours in your image](https://docs.opencv.org/3.4.6/df/d0d/tutorial_find_contours.html)

### Check if point inside contour

**_Ref:_** [stackoverflow: How to check if point is placed inside contour?](https://stackoverflow.com/questions/50670326/how-to-check-if-point-is-placed-inside-contour)

**_References:_** [OpenCV-Python Tutorials: Contours : More Functions](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html)

# OpenCV with other API

## Wand

### Wand -> OpenCV

```python
with WandImage(filename=source_file, resolution=(RESOLUTION,RESOLUTION)) as img:
    img.format        = 'png'
    # Fill image buffer with numpy array from blob
    img_buffer=numpy.asarray(bytearray(img.make_blob()), dtype=numpy.uint8)
image = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
```

Ref [stackoverflow: How to convert wand image object to open cv image (numpy array)](https://stackoverflow.com/questions/37015966/how-to-convert-wand-image-object-to-open-cv-image-numpy-array)

### OpenCV -> Wand

```python
image = cv2.imread(<image path>)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.transpose(image, (1, 0, 2))
```
