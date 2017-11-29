# Function
## Rotate Image
```C++
Point2f center(img_src.cols / 2.0, img_src.rows / 2.0);
Mat M = getRotationMatrix2D(center, degree, 1.0)
Rect box = RotatedRect(center, img_src.size(), degredd).boundingRect();
warpAffine(img_src, img_dst, M, box.size(), INTER_CUBIC, BORDER_CONSTANT, Scalar(128, 128, 128));
```