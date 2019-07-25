# Common Command

## Extract 1 frame from a vide

```shell
ffmpeg -ss <time> -i <input_video> -vframes 1 -q:v 2 output.jpg
```

**_Ref:_** [stackoverflow: How to extract 1 screenshot for a video with ffmpeg at a given time?](https://stackoverflow.com/questions/27568254/how-to-extract-1-screenshot-for-a-video-with-ffmpeg-at-a-given-time)

<!--  -->
<br>

---

<!--  -->

## Split video

[Not Recommend] This method result not accurate.

```shell
ffmpeg -ss <start_time> -t <duration> -i <intput_video> -vcodec copy -acodec copy output.mp4
```

**_Ref:_** [Split video file. FFmpeg](http://www.kompx.com/en/split-video-file-ffmpeg.htm)

**Accurate method**

```shell
ffmpeg -y -i <input_video> -ss <start_time> -to <end_time> -codec copy output.mp4
```

**_Ref:_** [简书: FFmpeg 精准时间切割视频文件](https://zhuanlan.zhihu.com/p/28008666)

**_References:_**

- [stackoverflow: Cut part from video file from start position to end position with FFmpeg [duplicate]](https://superuser.com/a/377407)
