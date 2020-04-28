---
title: "OpenCV的那些事-视频读写的问题"
last_modified_at: 2020-02-22
categories:
  - OpenCV
tags:
  - OpenCV
  - Python
---

OpenCV 处理视频的读写，部分操作还需要 FFmpeg 配合实现。

## :fallen_leaf:两个必要的工具

- FFmpeg
- mediainfo

## :fallen_leaf:查看信息

## :fallen_leaf:读视频

### 基本读取

### 交错视频读取

## :fallen_leaf:写视频

### 控制码率

OpenCV 应该是有一套默认的编码器参数，在写视频的接口中并没有任何关于控制编码器参数的函数，因此我们采用 FFmpeg 实现码率可控的视频的写入。

**核心思想：** 将 OpenCV 的图像数据通过 PIL 转化成特定的内存存储格式，之后开管道将数据传递给 FFmpeg，由 FFmpeg 实现写视频。

```python
import subprocess
import cv2
from PIL import Image


class VideoWriter(object):
    def __init__(self,
                 out_path,
                 video_w,
                 video_h,
                 fps=25,
                 encoding='H264',
                 video_bitrate='11M',
                 mode='ffmpeg'):
        self._mode = mode
        self._w = video_w
        self._h = video_h
        self._fps = fps
        if mode == 'opencv':
            self._video = cv2.VideoWriter(out_path,
                                          cv2.VideoWriter_fourcc(*encoding),
                                          self._fps, (self._w, self._h))
        else:
            self._pipeline = subprocess.Popen(
                'ffmpeg -loglevel warning -y -f image2pipe -vcodec png -r {fps} -i - -vcodec h264 -profile:v high -level:v 5 -refs 6 -q:v 0 -r {fps} -b:v {bitrate} -pix_fmt yuv420p {out_path}'
                .format(fps=self._fps,
                        bitrate=video_bitrate,
                        out_path=out_path),
                stdin=subprocess.PIPE,
                shell=True)

    def write_frame(self, frame):
        assert (frame.shape[1] == self._w and frame.shape[0] == self._h)
        if self._mode == 'opencv':
            self._video.write(frame)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame.save(self._pipeline.stdin, 'PNG')
            # Use 'PNG' format will keep color better

    def release(self):
        if self._mode == 'opencv':
            self._video.release()
        else:
            self._pipeline.stdin.close()
            self._pipeline.wait()
```

**注意:** 参考链接 1 中是使用 PIL 的`JPEG`的格式写入图片数据到管道的，但实际使用中发现采用`JPEG`的格式会导致输出的视频图片帧颜色和原始的视频帧图像颜色发生变化，这里建议采用`PNG`的格式写入。

**_References:_**

- [stackoverflow: opencv - videowriter control bitrate](https://stackoverflow.com/a/42602576/4636081)
- [Blog: JPEG Image Quality in PIL](https://jdhao.github.io/2019/07/20/pil_jpeg_image_quality/)

## :fallen_leaf:准换视频封装格式
