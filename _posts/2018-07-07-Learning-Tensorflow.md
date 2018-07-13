# Graphs

> Tensorflow separates definition of computations from their execution.

## How tensorflow run

1. Build a graph
2. Use a session to execute operations in the graph

## What's a tensor

**An n-dimensional matrix**

- 0-d tensor: scalar(number)
- 1-d tensor: vector
- 2-d tensor: matrix
- ...

In computation graph:

- Nodes is: operators, variables or constants

- Edges: tensors

## Why graphs

1. Save computation. Only run subgraphs that lead to the values you want to fetch.
2. Break computation into small, differential pieces to facilitate auto-differentiation.
3. Facilitate distributed computation, spread the work across multiple CPUs, GPUs, TPUs or other devices.

# Constants

```python
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
```

# Functions

## `tf`

- **tf.Session()**

- **tf.Graph()**

- **tf.InteractiveSession()**
    A Tensorflow `Session` for use in interactive contexts, such as a shell.
    ```python
    sess = tf.InteractiveSession()
    a = tf.constant(5)
    b = tf.constant(6)
    c = a * b
    print(c.eval())
    ```

- **Create constant tensor**
    - `tf.zeros(shape, dtype=tf.float32, name=None)`

    - `tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True)`
    - `tf.fill(dims, value, name=None)`: Create a tensor filled with a scalar value

- **Random**
    - `tf.set_random_seed(seed)`: Set the graph-level random seed. You can use it after your build a graph and before run a session.
    - `tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)`
    - `tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)`
    - `tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)`
    - `tf.random_shuffle(value, seed=None, name=None)`

- **Variable**
    **All variable in Tensorflow need to be initialized!** It means that when you run a session, you need to init the variable before doing other ops. A good way is to use `tf.global_variables_initializer()`
    ```python
    # Create two variables.
    weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="weights")
    biases = tf.Variable(tf.zeros([200]), name="biases")
    ...
    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()
    # Later, when launching the model
    with tf.Session() as sess:
        # Run the init operation.
        sess.run(init_op)
        ...
        # Use the model
        ...
    ```
    - Init a single variable
        ```python
        a = tf.Variable(tf.zeros([10, 10]))
        with tf.Session() as sess:
            sess.run(a.initializer)
            print(a.eval())
        ```
    - `tf.Variable.assign()`

- **Placeholder**
    > We can later supply data when we need to execute the computation

    When run the session, we need to feed value to the placeholder
    ```python
    a = tf.placeholder(tf.float32, shape=[3])
    b = tf.constant([5, 5, 5], tf.float32)
    c = a + b
    with tf.Session() as sess:
        # feed [1,2,3] to placeholder a via the dict {a: [1, 2, 3]}
        print(sess.run(c, {a: [1, 2, 3]}))
    ```
    Feed value once at a time, such as one image for a epoch
    ```python
    with tf.Session() as sess:
        for a_value in list_of_values_for_a:
            print(sess.run(c, feed_dict={a: a_value}))

- **tf.train.Saver**
    **Save graph's variables in binary files**
    Save parameters after 1000 steps
    ```python
    # create a saver object
    saver = tf.train.Saver()
    # launch a session to compute the graph
    with tf.Session as sess:
        # actual training loop
        for step in range(training_steps):
            sess.run([optimizer])
            if (step + 1) % 1000 == 0:
                saver.save(sess, <save_path>, global_step=model.global_step)
    ```

    ***References***
    - [Tensorflow API: Constants, Sequences, and Random Values](https://www.tensorflow.org/api_guides/python/constant_op#Random_Tensors)
    - [Tensorflow Programmers's Guide: Variables: Creation, Initialization, Saving, and Loading](https://www.tensorflow.org/versions/r1.0/programmers_guide/variables)


- **tf.slice**: Extracts a slice from a tensor.

# tf.summary and Tensorboard

Start Tensorboard

```shell
tensorboard --logdir=<tensorflow run log path> [--port]
```

# Websites

- [Stanford CS 20: Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/syllabus.html)


# tf.nn, tf.layers, tf.contrib区别

***References:***

- [小鹏的专栏: tf API 研读1：tf.nn，tf.layers， tf.contrib概述](https://cloud.tencent.com/developer/article/1016697)

# Data formats

- `N`
- `H`
- `W`
- `C`

- `NCHW` or `channels_first`
- `NHWC` or `channels_last`

`NHWC` is the TensorFlow default and `NCHW` is the optimal format to use when training on NVIDIA GPUs using cuDNN.

***References:***

- [Tensorflow doc: Performance Guide](https://www.tensorflow.org/performance/performance_guide)

# API

## `tf`

- `tf.reverse`: Reverse data in specific/given axis
    Application: When using OpenCV read image as BRG, `tf.reverse` can convert it into RGB
    ```python
    output = tf.reverse(img_tensor, [-1])
    ```

- `tf.agrmax(ingput, axis=None, ...)`: Returns the index with the largest value across axes of a tensor.

- `tf.get_collection`: Get a list of `Variable` from a collection
    ***References:***

    - [Blog: 【TensorFlow动手玩】常用集合: Variable, Summary, 自定义](https://blog.csdn.net/shenxiaolu1984/article/details/52815641)

- `tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)`: Clips tensor values to a specified min and max
    ```python
    a = np.array([[1,1,2,4], [3,4,5,8]])
    with tf.Session() as sess:
        print(sess.run(tf.clip_by_value(a, 2, 5)))
    >>> [[2 2 2 4]
         [3 4 5 5]]
    ```

- `tf.extract_image_patches(images, ksizes, strides, rates, padding, name=None)`: Extract `patches` from `images` and put them in the "depth" output dimension.

    ***References:***
    - [知乎: 关于tf.extract_image_patches的一些理解](https://zhuanlan.zhihu.com/p/37077403)

- `tf.reduce_sum(input_tensor, axis=None, ...)`: Computes the sum of elements across dimensions of a tensor.

- `tf.split(value, num_or_size_splits, axis=0, ...)`: Splits a tensor into sub tensors.
    ```python
    # Input: image, mask both are (h, w, 3)
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)
    #
    batch_raw, masks_raw = tf.split(input_image, 2, axis=2)
    ```

## 'tf.nn'

- `tf.nn.conv2d(input, filter, strides, padding, ...)`: Computes a 2-D convolution given 4-D `input` and `filter` tensors.

- `tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding='SAME', data_format='NHWC', name=None)`: The transpose of conv2d

    **理解:** transpose convolution相当于一个在周围/中间进行了padding之后的卷积，本质上还是卷积，只不过由于在卷积前进行了padding，所提使得输出的图像大小增加了。See gif from [here](https://github.com/vdumoulin/conv_arithmetic).

***References:***

- [知乎: 关于tf中的conv2d_transpose的用法](https://zhuanlan.zhihu.com/p/31988761)
- [简书: 理解tf.nn.conv2d和tf.nn.conv2d_transpose](https://www.jianshu.com/p/a897ed29a8a0)
- [StackExchange: What are deconvolutional layers?](https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers)

## `tf.layer`

- `tf.layer.conv2d(inputs, filters, kernel_size, strides=(1, 1), ...)`: Functional interface for the 2D convolution layer.

    **Note:** Pay a attention to the difference between `tf.nn.conv2d` and `tf.layer.conv2d`: `tf.nn.conv2d` is more basic, `filter` in it is `tensor`. Is calculate `input` and `filter` convlution. While `filter` in `tf.layer.conv2d` is a `int` number, and it creates tensor `filter` then does convolution calculation.

## `tf.data`


***References:***

- [知乎: TensorFlow全新的数据读取方式：Dataset API入门教程](https://zhuanlan.zhihu.com/p/30751039)

## Save and Restore model

***References:***
- [CV-Tricks.com: A quick complete tutorial to save and restore Tensorflow models](http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)