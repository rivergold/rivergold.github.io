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

## gRPC for Python

Example:

***References:***

- [Github grpc/grpc: example/python/helloworld](https://github.com/grpc/grpc/tree/v1.18.0/examples/python/helloworld)
