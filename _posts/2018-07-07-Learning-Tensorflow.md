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

A example shows how to write `tf.summary`: [Blog: Tensorflow学习笔记——Summary用法](https://www.cnblogs.com/lyc-seu/p/8647792.html)

# Websites

- [Stanford CS 20: Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/syllabus.html)


# tf.nn, tf.layers, tf.contrib区别

***References:***

- [小鹏的专栏: tf API 研读1：tf.nn，tf.layers， tf.contrib概述](https://cloud.tencent.com/developer/article/1016697)



# Tensorflow

## Graphs and Sessions

**tf.Graph** consists of **node (`tf.Operation`)** and **edge (`tf.Tensor`)**.

**tf.Session** class represents a connection between the client program (your code write with Python or similar interface available in other languages) and and the C++ runtime. 
A `tf.Session` object provides access to devices in the local machine, and remote devices using the distributed TensorFlow runtime.

<p align="center"> 
    <img src="http://ovvybawkj.bkt.clouddn.com/Tensorflow-Graph_and_Session.png">
</p>

***References:***

- [TensorFlow: 编程人员指南 *图表与会话*](https://www.tensorflow.org/programmers_guide/graphs?hl=zh-cn)

> TensorFlow 使用 tf.Session 类来表示客户端程序（通常为 Python 程序，但也提供了其他语言的类似接口）与 C++ 运行时之间的连接。tf.Session 对象使我们能够访问本地机器中的设备和使用分布式 TensorFlow 运行时的远程设备。它还可缓存关于 tf.Graph 的信息，使您能够多次高效地运行同一计算。

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

# Variable

## `name_scope` and `variable_scope`

***References:***

- [知乎: tensorflow里面name_scope, variable_scope等如何理解？](https://www.zhihu.com/question/54513728)
- [Blog: TensorFlow入门（七） 充分理解 name / variable_scope](https://blog.csdn.net/Jerr__y/article/details/70809528)
- [Tensorflow Guide: Variable](https://www.tensorflow.org/guide/variables)

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

- `tf.set_random_seed`

    ***References:***

    - [TensorFlow api: tf.set_random_seed](https://www.tensorflow.org/api_docs/python/tf/set_random_seed)

- `tf.py_func()`: Call Python code in Tensorflow graph
    ***References:***
    - [TensorFlow Guide Importing Data: Applying arbitrary Python logic with tf.py_func()](https://www.tensorflow.org/guide/datasets)

## `tf.nn`

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

- `tf.data.Dataset`

- `tf.data.Iterator`

***References:***

- [知乎: TensorFlow全新的数据读取方式：Dataset API入门教程](https://zhuanlan.zhihu.com/p/30751039)

## `tf.estimator`

### `tf.estimator.Estimator`

Estimator will automatically write the following to disk:

- checkpoints, which are version of the model created during training
- event files, which contain information that **TensorBoard** uses to create visualizations

#### `model_fn`

A [good example](https://www.epubit.com/selfpublish/article/1156;jsessionid=63E557268B23BE8DE6E71F3AFDACD4B0) to write `model_fn` for `tf.estimator.Estimator` 

### `tf.estimator.RunConfig`

```python
estimator_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 2 * 60 # Save checkpoints every 20 minutes
    keep_checkpoint_max = 10, # Retain the 10 most recent checkpoints
    )
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='models/iris',
    config=my_checkpointing_config)
```

***References:***

- [TensorFlow Guide: Checkpoints](https://www.tensorflow.org/guide/checkpoints#checkpointing_frequency)

**Methods:**

- `train`
    ```python
    train(input_fn,
          hooks=None,
          steps=None,
          max_steps=None,
          save_listeners=None)
    ```

***References:***

- [TensorFlow API: tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#train)

#### **Creating custom estimator**

***References:***

- [Tensorflow Guide: Creating Custom Estimators](https://www.tensorflow.org/guide/custom_estimators)
- [Github: tensorflow/models/samples/core/get_started/custom_estimator.py](https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py)

## `tf.train`

- `tf.train.get_global_step`: Get the global step tensor.

<!--  -->
<br>

***
<!--  -->

# Tricks

## TFRecord

1. Save data as `TFRecord` into disk
    <p align="center"> 
        <img src="http://ovvybawkj.bkt.clouddn.com/TF-Read-Data.png">
    </p>

2. Read data from `TFRecord` into `tf.data.Dataset`

[Why use TFRecord?](https://www.quora.com/What-are-the-benefits-of-using-TFRecord-files)

***References:***

- [知乎: YJango：TensorFlow中层API Datasets+TFRecord的数据导入](https://zhuanlan.zhihu.com/p/33223782)
- [Daniil's blog: Tfrecords Guide](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)
- [Stackoverflow: What are the advantages of using tf.train.SequenceExample over tf.train.Example for variable length features?](https://stackoverflow.com/questions/45634450/what-are-the-advantages-of-using-tf-train-sequenceexample-over-tf-train-example)

## Save and Restore model

TensorFlow provides two model formats:

- checkpoints, which is a format dependent on the code that created the model.
- SavedModel, which is a format independent of the code that created the model.

***References:***

- [TensorFlow: Guide *Checkpoints*](https://www.tensorflow.org/guide/checkpoints)
- [Tensorflow: Guide *Save and Restore*](https://www.tensorflow.org/guide/saved_model)
- [CV-Tricks.com: A quick complete tutorial to save and restore Tensorflow models](http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)

### Restore model weights from `checkpoint`

One way is: write your model, and then:

```python
vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
assign_ops = []
for var in vars_list:
    print(var.name)
    vname = var.name
    var_value = tf.contrib.framework.load_variable(<model_path>, vname)
    assign_ops.append(tf.assign(var, var_value))
sess.run(assign_ops)
```

**Note:** The `model_path` is a folder contain `checkpoint`.

## Load data

### `tf.data.Dataset`

1. Construct a `Dataset` from some tensors in memory, you can use `tf.data.Dataset.from_tensors()` or `tf.data.Dataset.from_tensor_slices()`.
2. If your input data are on disk in the recommended TFRecord format, you can construct a `tf.data.TFRecordDataset`.

***References:***

- [Blog: tensorflow中的dataset](http://d0evi1.com/tensorflow/datasets/)
- [Towards Data Science: Epoch vs Batch Size vs Iterations](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)

### Read imge

```python
img_string = tf.read_file(<img_path>)
img_decoded = tf.image.decode_jpeg(img_string)
```

**Note:** TensorFlow decode image into RGB, it is different with OpenCV.

- `tf.image.decode_image`: not give the image shape
- `tf.image.decode_jpeg` and `tf.image.decode_png` will give the image shape

***References:***

- [Github tensorflow/tensorflow Issue: tf.image.decode_image doesn't return tensor's shape #8551](https://github.com/tensorflow/tensorflow/issues/8551)
- [stackoverflow: TensorFlow:ValueError: 'images' contains no shape](https://stackoverflow.com/questions/44942729/tensorflowvalueerror-images-contains-no-shape)

***References:***

- [TensorFlow API tf.image.decode_jpeg](https://www.tensorflow.org/api_docs/python/tf/image/decode_jpeg)

### Process image with OpenCV in TensorFLow

```python
def process():
    pass
x = tf.py_func(process, <input>, <data stype>)
```

## Using `tf.estimator.Estimator` train / fine-tune model

### Notes

- `tf.estimator.Estimator` will auto restore the last checkpoint from the `model_dir` when you restart it. 

### Load a part of model from checkpoint

There are two ways:

- Using `RestoreHook`
    ***References:***
    - [Github tensorflow/tensorflow Issues: Support init_from_checkpoint and warm start with Distribution Strategy #19958](https://github.com/tensorflow/tensorflow/issues/19958)

- Using `tf.train.init_from_checkpoint` in your `model_fn`
    ***References:***
    - [stackoverflow: Load checkpoint and finetuning using tf.estimator.Estimator](https://stackoverflow.com/questions/46423956/load-checkpoint-and-finetuning-using-tf-estimator-estimator?noredirect=1&lq=1)

- Using `tf.estimator.WarmStartSettings`
    ***References:***
    - [TensorFlow API tf.estimator.WarmStartSettings](https://www.tensorflow.org/api_docs/python/tf/estimator/WarmStartSettings)

### Using pre-trained model from TensorFlow Hub

***References:***

- [Medium: Using Inception-v3 from TensorFlow Hub for transfer learning](https://medium.com/@utsumuki_neko/using-inception-v3-from-tensorflow-hub-for-transfer-learning-a931ff884526)

### Initialise optimizer variables in TensorFlow

***References:***

- [stackoverflow: How to initialise only optimizer variables in Tensorflow?](https://stackoverflow.com/questions/41533489/how-to-initialise-only-optimizer-variables-in-tensorflow/45624533)

## Load Data with `tf.data.Dataset` 

### Using `tf.data.Dataset` how to feed into Session

***References:***

- [stackoverflow: How to use dataset in TensorFlow session for training](https://stackoverflow.com/questions/47577108/how-to-use-dataset-in-tensorflow-session-for-training)

### Load different dataset during train

***References:***

- [stackoverflow: How to use Tensorflow's tf.cond() with two different Dataset iterators without iterating both?](https://stackoverflow.com/questions/46622490/how-to-use-tensorflows-tf-cond-with-two-different-dataset-iterators-without-i)

<!--  -->
<br>

***
<!--  -->

# My Tensorflow Pipeline

Mainly using functions:

- `tf.data.Dataset`: Prepare data
- `tf.estimator.Estimator`: Build model and train, evaluate and predict.

## Training:

### Using Dataset

***References:***

- [Blog: Dataset API详解](http://yvelzhang.site/2017/11/03/Dataset%20API/)

Mainly using:

- `tf.estimator.Estimator` method `train`
- Functions in `tf.train`
- And add some hooks into `tf.estimator.Estimator.train`

***References:***

- [TensorFlow API tf.estimator.Estimator.train](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#train)
- [TensorFlow API: Python API Guides Training](https://www.tensorflow.org/api_guides/python/train)
- [TensorFLow API Python API Guides Training Hooks](https://www.tensorflow.org/api_guides/python/train#Training_Hooks)

E.g.

```python
# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)
mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])
```

### Change learning decay

Find a decay function from [TensorFlow API Python API Guide Training/Decaying the learning rate](https://www.tensorflow.org/api_guides/python/train#Decaying_the_learning_rate), and then set it into one Optimizers.

***References:***

- [TensorFlow API: tf.train.exponential_decay](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay)

### Load pre-trained model

***References:***

- [Github tensorflow/tensorflow Issues: Estimator should be able to partially load checkpoints #10155](https://github.com/tensorflow/tensorflow/issues/10155)

<!--  -->
<br>

***
<!--  -->

# Erros & Solutions

- `TensorFlow ValueError: Cannot feed value of shape (1, 64, 64, 3) for Tensor u'Placeholder:0', which has shape '(1, ?, ?, 1)'`
    It means that you feed wrong data shape into TensorFlow placeholder.

    ***References:***
    - [stackoverflow: TensorFlow ValueError: Cannot feed value of shape (64, 64, 3) for Tensor u'Placeholder:0', which has shape '(?, 64, 64, 3)'](https://stackoverflow.com/questions/40430186/tensorflow-valueerror-cannot-feed-value-of-shape-64-64-3-for-tensor-uplace)

- `TensorFlow: “Attempting to use uninitialized value” in variable initialization`
    It means that you run your sess without initialise Variables. May you run variable initialise and train/predict in two different session?
    ***References:***
    - [stackoverflow: TensorFlow: “Attempting to use uninitialized value” in variable initialization](https://stackoverflow.com/questions/44624648/tensorflow-attempting-to-use-uninitialized-value-in-variable-initialization/44630421)

- When import tensorflow, occur error `ImportError: libcublas.so.9.0: cannot open shared object file: No such file or director`
    It means that maybe you use wrong `cuda` version. You need to set `export LD_LIBRARY_PATH=<your cuda install path>:$LD_LIBRARY_PATH`

    ***References:***
    - [Github tensorflow/tensorflow Issue: ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory #15604](https://github.com/tensorflow/tensorflow/issues/15604)
    - [stackoverflow: ImportError: libcublas.so.9.0: cannot open shared object file](https://stackoverflow.com/questions/48428415/importerror-libcublas-so-9-0-cannot-open-shared-object-file)

- When do `x / 127.5 - 1` occur error `TypeError: unsupported operand type(s) for /: 'Tensor' and 'float'`
    Maybe your `x` is not `tf.float32`, you can try `x = tf.cast(x, tf.float32)`

- When using `tf.data.Dataset` occur error `Tensorflow GetNext() failed because the iterator has not been initialized`

    ***References:***
    - [stackoverflow: Tensorflow GetNext() failed because the iterator has not been initialized](https://stackoverflow.com/questions/48443203/tensorflow-getnext-failed-because-the-iterator-has-not-been-initialized)
    - [TensorFlow API make_initializable_iterator](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#make_initializable_iterator)

<!--  -->
<br>

***
<!--  -->

# QA

## Tensorflow `tf.identity()` do for what?

When I watch `Tensorflow Models/resnet_model.py`, it use `inputs = tf.identity(inputs, 'initial_conv')`. Here is a good explanation from [Zhihu](https://zhuanlan.zhihu.com/p/32540546)

***References:***

- [知乎: TensorFlow 的 Graph 计算流程控制](https://zhuanlan.zhihu.com/p/32540546)

## Set learning rate decay in Tensorflow

A common function is `tf.train.exponential_decay`. It use `global_step` to calculate learning rate. TensorFlow `Optimimizer.minimize` need a args `global_step`. `global_step` will add 1 each batch. So if you want to update learning rate in epoch, you need to set `decay_step=<step> * num_batch`

***References:***

- [知乎: Tensorflow中learning rate decay的奇技淫巧](https://zhuanlan.zhihu.com/p/32923584)
- [知乎: 使用Tensorflow过程中遇到过哪些坑？](https://www.zhihu.com/question/269968195)
- [stackoverflow: How to set adaptive learning rate for GradientDescentOptimizer?](https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer/33922859)
- [Blog: Tensorflow一些常用基本概念与函数（四）](https://www.cnblogs.com/wuzhitj/p/6648641.html)