---
title: "FFMPEG血泪史"
categories:
  - memo
tags:
  - ffmpeg
  - tool
  - memo
---

一部 FFmpeg 使用的血泪史

## Install

### Use Static Builds

This is the simplest and least error-prone method to install ffmpeg.

Get builds from [FFmpeg.ory](https://ffmpeg.org/download.html), and set your shell `PATH`.

### Install Pyhton OpenCV with FFMPEG and X264 and mp3 Support

Note that `pip install opencv-python` install OpenCV has some problems like not support x264. Better use following way to install OpenCV.

```shell
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
conda install opencv
```

**注意：** 使用上述命令安装支持读写 x264 的 OpenCV，之后再删除通过 conda 安装的 ffmpeg，改为静态库版本。因为通过 conda 安装的 ffmpeg 不支持 mp3 的读写。

**_Ref_** [stackoverflow: How to use libx264 ffmpeg in conda environment?](https://superuser.com/questions/1420351/how-to-use-libx264-ffmpeg-in-conda-environment)

### Build from Source

Another way is to build ffmpeg from source.

**_Ref:_** [FFMPEG: Compile FFmpeg on CentOS](https://trac.ffmpeg.org/wiki/CompilationGuide/Centos)

You can build static or dynamic libs.

**:triangular_flag_on_post:注意：** 当编译动态库版本的 ffmpeg 时，其依赖也需要编译成动态库版本。如果需要编译动态库版本的 OpenCV，ffmpeg 也需要使用动态库。

**_Ref_** :thumbsup:[stackoverflow: FFmpeg doesn't compile with shared libraries](https://stackoverflow.com/questions/32785279/ffmpeg-doesnt-compile-with-shared-libraries)

#### Problem & Solution

##### libaom.a cannot be used when making a shared object

```shell
/usr/bin/ld: /home/rivergold/software/lib/ffmpeg/ffmpeg-build/lib/libaom.a(noise_model.c.o): relocation R_X86_64_PC32 against symbol `stderr@@GLIBC_2.2.5' can not be used when making a shared object; recompile with -fPIC
```

**_Solution:_**

Build libaom with enable shared libs.

```shell
# libaom
cd ${FFPEG_SOURCE_DIR} &&
    mkdir -p aom_build &&
    cd aom_build &&
    PATH="$PREFIX_DIR/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$PREFIX_DIR" -DBUILD_SHARED_LIBS=on -DENABLE_NASM=on ../aom &&
    PATH="$PREFIX_DIR/bin:$PATH" make &&
    make install
```

**注意：** [ffmpeg 官网的教程](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu)仅说明了如何编译 static 版本的 ffmpeg，且编译`libaom`的参数`ENABLE_SHARED`在 CMakeLists.txt 中没有记录，需要改为`BUILD_SHARED_LIBS`

## :fallen_leaf: Get Video Information

[mediainfo](https://mediaarea.net/en/MediaInfo/Download) is a very good tool to get media information. And [pymediainfo](https://github.com/sbraz/pymediainfo) is Python API of mediainfo.

## :fallen_leaf: Change Video Format

```shell
ffmpeg -i <in_video_path> -c copy <out_video_path>
```

E.g.

```shell
ffmpeg -i in.ts -c copy out.mp4
```

## :fallen_leaf: Change Video Encode Format

```shell
ffmpeg -i <in_video_path> -c:v <video_encode> -c:a copy <out_video_path>
```

E.g.

```shell
ffmpeg -i in.ts -c:v libx264 -c:a copy out.mp4
```

### Change Encode while Keep Bitrate

```shell
ffmpeg -i <in_video_path> -c:v <video_encode> -b:v <video_bitrate> -c:a copy <out_video_path>
```

E.g.

```shell
ffmpeg -i in.ts -c:v libx264 -b:v 20M -c:a copy out.mp4
```

**_Ref_** [stackoverflow: How to convert .ts file into a mainstream format losslessly?](https://askubuntu.com/questions/716424/how-to-convert-ts-file-into-a-mainstream-format-losslessly)

## Extrac Frame from Video

```shell
ffmpeg -ss <time> -i <input_video> -vframes 1 -q:v 2 output.jpg
```

**_Ref_** [stackoverflow: How to extract 1 screenshot for a video with ffmpeg at a given time?](https://stackoverflow.com/questions/27568254/how-to-extract-1-screenshot-for-a-video-with-ffmpeg-at-a-given-time)

## Cut Video

```shell
# start_time, end_time
ffmpeg -ss <start_time> -to <end_time> -i <in_video_path> <out_video_path>
# start_time, duration
ffmepg -ss <start_time> -t <duration> -i <in_video_path> <out_video_path>
```

<!-- ```shell
ffmpeg -ss <start_time> -t <duration> -i <intput_video> -vcodec copy -acodec copy output.mp4
```

**_Ref:_** [Split video file. FFmpeg](http://www.kompx.com/en/split-video-file-ffmpeg.htm) -->

<!-- **Accurate method**

```shell
ffmpeg -ss 32.920 -to 35.720  -i <input_video> -c:v libx264 <output_video_path>
```

**_Ref:_** [stackoverflow: How to cut at exact frames using ffmpeg?](https://superuser.com/a/459488) -->

<!-- ```shell
ffmpeg -y -i <input_video> -ss <start_time> -to <end_time> -codec copy output.mp4
```

**_Ref:_** [简书: FFmpeg 精准时间切割视频文件](https://zhuanlan.zhihu.com/p/28008666) -->

**_References:_**

- [stackoverflow: Cut part from video file from start position to end position with FFmpeg [duplicate]](https://superuser.com/a/377407)

---

## Get Video Frame Number

```shell
# Use ffmpeg
ffmpeg -i <video_path>  -vcodec copy -acodec copy -f null /dev/null 2>&1 | grep 'frame=' | cut -f 2 -d ' '
# Use ffprobe
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames <video_path>
```

**_References:_**

- [腾讯云: 如何用 ffmpeg 取帧数？](https://cloud.tencent.com/developer/ask/103796)
- [Quora: How do I calculate the number of all frames in a video, using FFmpeg?](https://qr.ae/TWvXzT)

---

## Get Video Information

```shell
ffprobe -print_format json <video>
```

**_References:_**

- [Github Gist: jaivikram/getVideoDetails.py](https://gist.github.com/jaivikram/4690569)

- [Github Gist: oldo/video-metada-finder.py](https://gist.github.com/oldo/dc7ee7f28851922cca09)

## Add audio into video

```shell
ffmpeg -i <audio_path> -i <video_path> -codec copy <output_path>
```

## Set Log Level

```shell
{ "quiet"  , AV_LOG_QUIET   },
{ "panic"  , AV_LOG_PANIC   },
{ "fatal"  , AV_LOG_FATAL   },
{ "error"  , AV_LOG_ERROR   },
{ "warning", AV_LOG_WARNING },
{ "info"   , AV_LOG_INFO    },
{ "verbose", AV_LOG_VERBOSE },
{ "debug"  , AV_LOG_DEBUG   },
{ "trace"  , AV_LOG_TRACE   },
```

```shell
ffmpeg -loglevel warning ...
```

**_References:_**

- [简书: ffmpeg # 利用 loglevel 控制打印日志的信息](https://www.jianshu.com/p/2be79f17e271)
- [stackoverflow: How can I make ffmpeg be quieter/less verbose?](https://superuser.com/questions/326629/how-can-i-make-ffmpeg-be-quieter-less-verbose)

## Write Text into Video

**_Ref_** [stackoverflow: Using hex colors with ffmpeg's showwaves](https://stackoverflow.com/questions/45885460/using-hex-colors-with-ffmpegs-showwaves)

## Add Image into Video

**_Ref_** [stackoverflow: Add an image overlay in front of video using ffmpeg](https://video.stackexchange.com/questions/12105/add-an-image-overlay-in-front-of-video-using-ffmpeg)

---

## Change Video Resolution

```shell
ffmpeg -i <in_video_path> -vf scale=<new_w>:<new_h> <out_video_path>
```

**_Ref_** [Blog: 使用 ffmpeg 修改视频文件的分辨率](https://blog.p2hp.com/archives/5512)

## Concat

### Concat Video

**_Ref_** [stackoverflow: How to concatenate two MP4 files using FFmpeg?](https://stackoverflow.com/a/11175851/4636081)

### Concat Audio

```shell
ffmpeg -i "concat:<audio_1>|<audio_2>" -acodec copy <output_audio>
```

**_Ref:_** [stackoverflow: How to join/merge many mp3 files?](https://superuser.com/a/314245)

## Replace Video Audio

```shell
# Erase video raw audio
ffmpeg -i <input_video> -codec copy -an <output_video>
# Add new audio
ffmpeg -i <audio_path> -i <video_path> -codec copy -shortest <output_path>
```

**_Ref_** [Blog: Replacing video audio using ffmpeg](https://ochremusic.com/2016/07/05/replacing-video-audio-using-ffmpeg/)

## :fallen_leaf:Problems & Solutions

### Variable FPS

Use `mediainfo` to check if the video has variable fps.

**_Ref_** [VideoHelp Forum: Constant frame rate with H264 ts stream
](https://forum.videohelp.com/threads/365853-Constant-frame-rate-with-H264-ts-stream)

### 视频交错问题

在转换视频编码格式为 h264 时，转换后的视频在文字、人体边缘等位置出现横线、锯齿的问题，即所谓的*拉丝效应*。添加`-deinterlace`说明需要进行反交错处理，可以解决该问题。具体的原因还有待进一步探究。

## :fallen_leaf:Good FFmpeg Command Collection

- [Github Gist protrolium/ffmpeg.md](https://gist.github.com/protrolium/e0dbd4bb0f1a396fcb55)
