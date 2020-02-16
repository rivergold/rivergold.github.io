***References:***

- [简书: grpc| python 实战 grpc](https://www.jianshu.com/p/43fdfeb105ff)

# Protobuf

## Bacis

### Example of `.proto`

```protobuf
syntax = "proto3";

package face;

service Face{
    rpc DetectFace(Request) returns(Result) {}
}

message Request{
    uint32 img_w = 1;
    uint32 img_h = 2;
    bytes img_data = 3;
}

message Result{
    string message = 1;
}
```

### What does `1`, `2`, `3` mean in protobuf

***References:***

- [stackoverflow: What does “1”, “2”, “3” mean in protobuf?](https://stackoverflow.com/a/9018294/4636081)

## Tricks

### numpy to protobuf

***References:***

- [stackoverflow: Decoding Image from Protobuf using Python](https://stackoverflow.com/questions/44060763/decoding-image-from-protobuf-using-python)

### C++ OpenCV Mat into protobuf

***References:***

- [stackoverflow: How can I serialize cv::Mat objects using protobuf?](https://stackoverflow.com/questions/51553943/how-can-i-serialize-cvmat-objects-using-protobuf)

<!--  -->
<br>

***

<br>
<!--  -->

# gRPC

**At first, you need to read [What is gRPC](https://grpc.io/docs/guides/).**

gRPC Stream

***References:***

- [segmentfault: 带入gRPC：gRPC Streaming, Client and Server](https://segmentfault.com/a/1190000016503114)

gRPC异步

***References:***

- [Senlin's Blog: 谈谈 gRPC 的 C++ 异步编程](http://senlinzhan.github.io/2017/08/10/grpc-async/)

## gRPC for Python

### Install

```bash
pip install grpcio-tools
```

Ref [gRPC Basics - Python](https://grpc.io/docs/tutorials/basic/python.html#generating-client-and-server-code).

**Note:** It is best to use `pip` instead of `conda` to install, because the version from `conda` is lower than `pip`.

### Example

***References:***

- [Github grpc/grpc: example/python/helloworld](https://github.com/grpc/grpc/tree/v1.18.0/examples/python/helloworld)

### Tricks

#### Async Client

***References:***

- [Github grpc/grpc how to create an async client use python #16329](https://github.com/grpc/grpc/issues/16329)
