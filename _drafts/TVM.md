# Concept

> 基于Halide思想，将算法的算法的数学表达与计算顺序（这里的计算顺序我的理解是计算的实现）分离。计算图表示了算法的数学表达；使用schedule实现算法的计算顺序。

## Schedule

> Using existing programming tools, writing high-performance image processing code requires sacrificing readability, portability, and
modularity. We argue that this is a consequence of conflating what
computations define the algorithm, with decisions about storage
and the order of computation. We refer to these latter two concerns
as the schedule, including choices of tiling, fusion, recomputation
vs. storage, vectorization, and parallelism.

## Others

### NEON

[Github: NervanaSystems/neon](https://github.com/NervanaSystems/neon)

neon is Intel's reference deep learning framework committed to best performance on all hardware. Designed for ease-of-use and extensibility.

# Linux

## List all sy

```bash
ls -la /var/www/ | grep "\->"
```

Ref [StackExchange: How to list all symbolic links in a directory](https://askubuntu.com/questions/522051/how-to-list-all-symbolic-links-in-a-directory)