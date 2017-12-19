# Function
## Rotate Image
- `cv::getRotationMatrix2D(Poiont2f center, double angle double scale)`
- `cv::warpAffine`

**C++**
```C++
Mat img_dst;
Point2f center(img_src.cols / 2.0, img_src.rows / 2.0);
Mat R = getRotationMatrix2D(center, degree, 1.0);
warpAffine(img_src, img_dst, R, img_src.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128, 128, 128));
```

**Python**
```python  
center = (img.shape[1] / 2, img.shape[0] / 2)
degree = 30
R = cv2.getRotationMatrix2D(center, degree, 1)
dst = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]), 
                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, 
                     borderValue=(128,128,128))
```

These method above will crop image as the same size of the input image. The following will show how to **do rotate without crop:**<br>

**C++**
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

**Python**
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

***References:***
- [OpenCV doc OpenCV-Python Tutorials: Image Processing in OpenCV: Geometric Transformations of Images](https://docs.opencv.org/3.2.0/da/d6e/tutorial_py_geometric_transformations.html)
- [OpenCV doc: warpAffine](https://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983)
- [stackoverflow: Rotate an image without cropping in OpenCV in C++](https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c)