# Good Blog

- [Github Gist protrolium/ffmpeg.md](https://gist.github.com/protrolium/e0dbd4bb0f1a396fcb55)

# Common Command

## Convert video type

### Convert into H264

```shell
ffmpeg -i <input video> -c:v libx264 -c:a copy <output.mp4>
```

**_Ref:_** [stackoverflow: How to convert .ts file into a mainstream format losslessly?](https://askubuntu.com/questions/716424/how-to-convert-ts-file-into-a-mainstream-format-losslessly)

---

## Extract 1 frame from a vide

```shell
ffmpeg -ss <time> -i <input_video> -vframes 1 -q:v 2 output.jpg
```

**_Ref:_** [stackoverflow: How to extract 1 screenshot for a video with ffmpeg at a given time?](https://stackoverflow.com/questions/27568254/how-to-extract-1-screenshot-for-a-video-with-ffmpeg-at-a-given-time)

---

## Split video

[Not Recommend] This method result not accurate.

```shell
ffmpeg -ss <start_time> -t <duration> -i <intput_video> -vcodec copy -acodec copy output.mp4
```

**_Ref:_** [Split video file. FFmpeg](http://www.kompx.com/en/split-video-file-ffmpeg.htm)

**Accurate method**

```shell
ffmpeg -ss 32.920 -to 35.720  -i <input_video> -c:v libx264 <output_video_path>
```

**_Ref:_** [stackoverflow: How to cut at exact frames using ffmpeg?](https://superuser.com/a/459488)

<!-- ```shell
ffmpeg -y -i <input_video> -ss <start_time> -to <end_time> -codec copy output.mp4
```

**_Ref:_** [简书: FFmpeg 精准时间切割视频文件](https://zhuanlan.zhihu.com/p/28008666) -->

**_References:_**

- [stackoverflow: Cut part from video file from start position to end position with FFmpeg [duplicate]](https://superuser.com/a/377407)

---

## Get total frame number of a video

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

---

## Set log level

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

---

## Write text into video

**_References:_**

- [stackoverflow: Using hex colors with ffmpeg's showwaves](https://stackoverflow.com/questions/45885460/using-hex-colors-with-ffmpegs-showwaves)

---

## Add image into video

***References:***

- [stackoverflow: Add an image overlay in front of video using ffmpeg](https://video.stackexchange.com/questions/12105/add-an-image-overlay-in-front-of-video-using-ffmpeg)

---

## Scale video resolution

```shell
ffmpeg -i <in_video_path> -vf scale=<to_w>:<to_h> <out_video_path>
```

***References:***

- [Blog: 使用ffmpeg修改视频文件的分辨率](https://blog.p2hp.com/archives/5512)

<!--  -->
<br>

---

<br>
<!--  -->

# Build FFMPEG on CentOS

Download source code from [FFmpeg.org](https://ffmpeg.org/download.html)

**_Ref:_** [FFMPEG: Compile FFmpeg on CentOS](https://trac.ffmpeg.org/wiki/CompilationGuide/Centos)

## Install via conda

***References:***

- [stackoverflow: How to use libx264 ffmpeg in conda environment?](https://superuser.com/questions/1420351/how-to-use-libx264-ffmpeg-in-conda-environment)


<!--  -->
<br>

---

<br>
<!--  -->

# Problems & Solutions

## Variable FPS

Use `mediainfo` to check if the video has variable fps.

- [mediainfo](https://mediaarea.net/en/MediaInfo/Download)

**_Ref:_** [VideoHelp Forum: Constant frame rate with H264 ts stream
](https://forum.videohelp.com/threads/365853-Constant-frame-rate-with-H264-ts-stream)
