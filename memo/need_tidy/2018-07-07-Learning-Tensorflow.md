# Graphs and Sessions

**tf.Graph** consists of **node (`tf.Operation`)** and **edge (`tf.Tensor`)**.

**tf.Session** class represents a connection between the client program (your code write with Python or similar interface available in other languages) and and the C++ runtime.
A `tf.Session` object provides access to devices in the local machine, and remote devices using the distributed TensorFlow runtime.

<p align="center"> 
    <img src="http://ovvybawkj.bkt.clouddn.com/Tensorflow-Graph_and_Session.png">
</p>

**_References:_**

- [TensorFlow: 编程人员指南 _图表与会话_](https://www.tensorflow.org/programmers_guide/graphs?hl=zh-cn)

> TensorFlow 使用 tf.Session 类来表示客户端程序（通常为 Python 程序，但也提供了其他语言的类似接口）与 C++ 运行时之间的连接。tf.Session 对象使我们能够访问本地机器中的设备和使用分布式 TensorFlow 运行时的远程设备。它还可缓存关于 tf.Graph 的信息，使您能够多次高效地运行同一计算。

# Data formats

- `N`
- `H`
- `W`
- `C`

- `NCHW` or `channels_first`
- `NHWC` or `channels_last`

`NHWC` is the TensorFlow default and `NCHW` is the optimal format to use when training on NVIDIA GPUs using cuDNN.

**_References:_**

- [Tensorflow doc: Performance Guide](https://www.tensorflow.org/performance/performance_guide)

# Variable

## `name_scope` and `variable_scope`

**_References:_**

- [知乎: tensorflow 里面 name_scope, variable_scope 等如何理解？](https://www.zhihu.com/question/54513728)
- [Blog: TensorFlow 入门（七） 充分理解 name / variable_scope](https://blog.csdn.net/Jerr__y/article/details/70809528)
- [Tensorflow Guide: Variable](https://www.tensorflow.org/guide/variables)

## API

### tf

#### `tf.reverse`: Reverse data in specific/given axis

    Application: When using OpenCV read image as BRG, `tf.reverse` can convert it into RGB
    ```python
    output = tf.reverse(img_tensor, [-1])
    ```

#### `tf.agrmax(ingput, axis=None, ...)`: Returns the index with the largest value across axes of a tensor.

#### `tf.get_collection`: Get a list of `Variable` from a collection

    ***References:***

    - [Blog: 【TensorFlow动手玩】常用集合: Variable, Summary, 自定义](https://blog.csdn.net/shenxiaolu1984/article/details/52815641)

#### `tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)`: Clips tensor values to a specified min and max

    ```python
    a = np.array([[1,1,2,4], [3,4,5,8]])
    with tf.Session() as sess:
        print(sess.run(tf.clip_by_value(a, 2, 5)))
    >>> [[2 2 2 4]
         [3 4 5 5]]
    ```

#### `tf.extract_image_patches(images, ksizes, strides, rates, padding, name=None)`: Extract `patches` from `images` and put them in the "depth" output dimension.

    ***References:***
    - [知乎: 关于tf.extract_image_patches的一些理解](https://zhuanlan.zhihu.com/p/37077403)

#### `tf.reduce_sum(input_tensor, axis=None, ...)`: Computes the sum of elements across dimensions of a tensor.

- `tf.split(value, num_or_size_splits, axis=0, ...)`: Splits a tensor into sub tensors.
  ```python
  # Input: image, mask both are (h, w, 3)
  image = np.expand_dims(image, 0)
  mask = np.expand_dims(mask, 0)
  input_image = np.concatenate([image, mask], axis=2)
  #
  batch_raw, masks_raw = tf.split(input_image, 2, axis=2)
  ```

#### `tf.set_random_seed`

    ***References:***

    - [TensorFlow api: tf.set_random_seed](https://www.tensorflow.org/api_docs/python/tf/set_random_seed)

#### `tf.py_func()`: Call Python code in Tensorflow graph

**_References:_**

- [TensorFlow Guide Importing Data: Applying arbitrary Python logic with tf.py_func()](https://www.tensorflow.org/guide/datasets)
- [stackoverflow: Output from TensorFlow `py_func` has unknown rank/shape](https://stackoverflow.com/questions/42590431/output-from-tensorflow-py-func-has-unknown-rank-shape)

Pass `string` into `tf.py_func`

```python
def _parse_func(file_name):
    # Read file
    return np.ndarray
dataset = tf.data.Dataset.from_tensor_slices((names_list))
dataset = dataset.map(lambda filename: tf.py_func(_parse_func, [filename], tf.float32))
```

**_References:_**

- [Github tensorflow/tensorflow: tf Dataset with tf.py_func doesn't work as the tutorial says #12396](https://github.com/tensorflow/tensorflow/issues/12396)

---

### tf.nn

#### `tf.nn.conv2d(input, filter, strides, padding, ...)`: Computes a 2-D convolution given 4-D `input` and `filter` tensors.

#### `tf.nn.conv2d_transpose(value, filter, output_shape, strides, padding='SAME', data_format='NHWC', name=None)`: The transpose of conv2d

**理解:** transpose convolution 相当于一个在周围/中间进行了 padding 之后的卷积，本质上还是卷积，只不过由于在卷积前进行了 padding，所提使得输出的图像大小增加了。See gif from [here](https://github.com/vdumoulin/conv_arithmetic).

**_References:_**

- [知乎: 关于 tf 中的 conv2d_transpose 的用法](https://zhuanlan.zhihu.com/p/31988761)
- [简书: 理解 tf.nn.conv2d 和 tf.nn.conv2d_transpose](https://www.jianshu.com/p/a897ed29a8a0)
- [StackExchange: What are deconvolutional layers?](https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers)

---

### tf.layer

#### `tf.layer.conv2d(inputs, filters, kernel_size, strides=(1, 1), ...)`: Functional interface for the 2D convolution layer.

**Note:** Pay a attention to the difference between `tf.nn.conv2d` and `tf.layer.conv2d`: `tf.nn.conv2d` is more basic, `filter` in it is `tensor`. Is calculate `input` and `filter` convlution. While `filter` in `tf.layer.conv2d` is a `int` number, and it creates tensor `filter` then does convolution calculation.

---

### tf.data

#### `tf.data.Dataset`

#### `tf.data.Iterator`

**_References:_**

- [知乎: TensorFlow 全新的数据读取方式：Dataset API 入门教程](https://zhuanlan.zhihu.com/p/30751039)

---

### tf.estimator

#### tf.estimator.Estimator

Estimator will automatically write the following to disk:

- checkpoints, which are version of the model created during training
- event files, which contain information that **TensorBoard** uses to create visualizations

#### `model_fn`

A [good example](https://www.epubit.com/selfpublish/article/1156;jsessionid=63E557268B23BE8DE6E71F3AFDACD4B0) to write `model_fn` for `tf.estimator.Estimator`

#### `tf.estimator.RunConfig`

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

**_References:_**

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

**_References:_**

- [TensorFlow API: tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#train)

#### **Creating custom estimator**

**_References:_**

- [Tensorflow Guide: Creating Custom Estimators](https://www.tensorflow.org/guide/custom_estimators)
- [Github: tensorflow/models/samples/core/get_started/custom_estimator.py](https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py)

### tf.train

- `tf.train.get_global_step`: Get the global step tensor.

<!--  -->
<br>

---

<!--  -->

# Tricks

## TFRecord

1. Save data as `TFRecord` into disk

   <p align="center"> 
       <img src="http://ovvybawkj.bkt.clouddn.com/TF-Read-Data.png">
   </p>

2. Read data from `TFRecord` into `tf.data.Dataset`

[Why use TFRecord?](https://www.quora.com/What-are-the-benefits-of-using-TFRecord-files)

**_References:_**

- [知乎: YJango：TensorFlow 中层 API Datasets+TFRecord 的数据导入](https://zhuanlan.zhihu.com/p/33223782)
- [Daniil's blog: Tfrecords Guide](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)
- [Stackoverflow: What are the advantages of using tf.train.SequenceExample over tf.train.Example for variable length features?](https://stackoverflow.com/questions/45634450/what-are-the-advantages-of-using-tf-train-sequenceexample-over-tf-train-example)

## Save and Restore model

TensorFlow provides two model formats:

- checkpoints, which is a format dependent on the code that created the model.
- SavedModel, which is a format independent of the code that created the model.

**_References:_**

- [TensorFlow: Guide _Checkpoints_](https://www.tensorflow.org/guide/checkpoints)
- [Tensorflow: Guide _Save and Restore_](https://www.tensorflow.org/guide/saved_model)
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

**_References:_**

- [Blog: tensorflow 中的 dataset](http://d0evi1.com/tensorflow/datasets/)
- [Towards Data Science: Epoch vs Batch Size vs Iterations](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)

### Read imge

```python
img_string = tf.read_file(<img_path>)
img_decoded = tf.image.decode_jpeg(img_string)
```

**Note:** TensorFlow decode image into RGB, it is different with OpenCV.

- `tf.image.decode_image`: not give the image shape
- `tf.image.decode_jpeg` and `tf.image.decode_png` will give the image shape

**_References:_**

- [Github tensorflow/tensorflow Issue: tf.image.decode_image doesn't return tensor's shape #8551](https://github.com/tensorflow/tensorflow/issues/8551)
- [stackoverflow: TensorFlow:ValueError: 'images' contains no shape](https://stackoverflow.com/questions/44942729/tensorflowvalueerror-images-contains-no-shape)

**_References:_**

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
  **_References:_**

  - [Github tensorflow/tensorflow Issues: Support init_from_checkpoint and warm start with Distribution Strategy #19958](https://github.com/tensorflow/tensorflow/issues/19958)

- Using `tf.train.init_from_checkpoint` in your `model_fn`
  **_References:_**

  - [stackoverflow: Load checkpoint and finetuning using tf.estimator.Estimator](https://stackoverflow.com/questions/46423956/load-checkpoint-and-finetuning-using-tf-estimator-estimator?noredirect=1&lq=1)

- Using `tf.estimator.WarmStartSettings`
  **_References:_**
  - [TensorFlow API tf.estimator.WarmStartSettings](https://www.tensorflow.org/api_docs/python/tf/estimator/WarmStartSettings)

### Using pre-trained model from TensorFlow Hub

**_References:_**

- [Medium: Using Inception-v3 from TensorFlow Hub for transfer learning](https://medium.com/@utsumuki_neko/using-inception-v3-from-tensorflow-hub-for-transfer-learning-a931ff884526)

### Initialise optimizer variables in TensorFlow

**_References:_**

- [stackoverflow: How to initialise only optimizer variables in Tensorflow?](https://stackoverflow.com/questions/41533489/how-to-initialise-only-optimizer-variables-in-tensorflow/45624533)

## Load Data with `tf.data.Dataset`

### Using `tf.data.Dataset` how to feed into Session

**_References:_**

- [stackoverflow: How to use dataset in TensorFlow session for training](https://stackoverflow.com/questions/47577108/how-to-use-dataset-in-tensorflow-session-for-training)

### Load different dataset during train

**_References:_**

- [stackoverflow: How to use Tensorflow's tf.cond() with two different Dataset iterators without iterating both?](https://stackoverflow.com/questions/46622490/how-to-use-tensorflows-tf-cond-with-two-different-dataset-iterators-without-i)

## Using `tf.train.Saver` to save and restore `checkpoint`

### Save

```python
# Build your computing graph
# ....
saver = tf.train.Saver()

# During training
for epoch in range(num_epoch):
    # ...
    saver.save(sess, epoch)
    # TODO
```

### Restore

```python
# Build your computing graph
# ....
saver = tf.train.Saver()
saver.restore(sess, <checkpont_path>)
```

**Note**

If you want to restore a `.ckpt` and continue to training, you should pay a attention to your learning rate decay. Because learning rate decay computation is related to `global_step`, you need set correct `global_step` to your learning reate decay.

**_References:_**

- []

### Print tensor name and value in `.ckpt`

```python
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# Print tensors name
print_tensors_in_checkpoint_file(file_name=<ckpt_path>, tensor_name='', all_tensor_names=True, all_tensors=False)
# Print tensors name and value
print_tensors_in_checkpoint_file(file_name=<ckpt_path>, tensor_name='', all_tensor_names=True, all_tensors=True)
```

**_References:_**

- [stackoverflow: How do I find the variable names and values that are saved in a checkpoint?](https://stackoverflow.com/questions/38218174/how-do-i-find-the-variable-names-and-values-that-are-saved-in-a-checkpoint)

## Save model as `.pb` and load without rebuild the network

### Save Model

```python
import tensorflow as tf
from tensorflow.python.save_model import tag_constants

sess = tf.Session()
... # Build your network
# Saving
inputs = {
    'x_placeholder': _x,
}
outputs = {'output': model_output}
tf.saved_model.simple_save(sess, 'path/to/your/location', inputs, outputs)
```

Your will see `saved_model.pb` and a folder `variables` in your path.

### Load Model and Predict

```python
import tensorflow as tf
from tensorflow.python.save_model import tag_constants

sess = tf.Session()
tf.saved_model.loader.load(sess, [tag_constants.SERVING], 'path/to/your/location')

_x = sess.graph.get_tensor_by_name('<placeholder name>')
_y = sess.graph.get_tensor_by_name('<output name>')

sess.run(_y, feed_dict={_x: <data>})
```

**Note:** When you load model, and `get_tensor_by_name(<name>)`, the `name` must be the `Placeholder name` or `Variable name`, like `_x = tf.placeholder(tf.float32, shape=[1, 1], name='x_placeholder')`.

For the output variable, if you don't knoe the name, you can use `_y = tf.identity(_y, name='y')`.

**_References:_**

- [stackoverflow: Tensorflow: how to save/restore a model?](https://stackoverflow.com/a/50852627/4636081)
- [stackoverflow: How to rename a variable which respects the name scope?](https://stackoverflow.com/a/34399966/4636081)
- [Medium Blog - MetaFLow: TensorFlow: How to freeze a model and serve it with a python API](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc)
- [TensorFLow Develop GUIDE: Save and Restore](https://www.tensorflow.org/guide/saved_model)

**TODO**

- [ ] How to freeze a model?

## Set GPU using size

### Set specific GPU devices

Here are two method:

```python
import os
os.environ('CUDA_VISIBLE_DEVICES') = '0'
```

or

```shell
CUDA_VISIBLE_DEVICES=0 python <your python script>
```

### Set fixed GPU memory for TensorFlow

```python
sess_config = tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=sess_config)
```

### Allow used GPU memory growth

```python
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
```

**_References:_**

- [stackoverflow: How to prevent tensorflow from allocating the totality of a GPU memory?](https://stackoverflow.com/a/48214084/4636081)
- [CSDN: Tensorflow 与 Keras 自适应使用显存](https://blog.csdn.net/l297969586/article/details/78905087)

<!--  -->
<br>

---

<!--  -->

# My Tensorflow Pipeline

Mainly using functions:

- `tf.data.Dataset`: Prepare data
- `tf.estimator.Estimator`: Build model and train, evaluate and predict.

## Training:

### Using Dataset

**_References:_**

- [Blog: Dataset API 详解](http://yvelzhang.site/2017/11/03/Dataset%20API/)

Mainly using:

- `tf.estimator.Estimator` method `train`
- Functions in `tf.train`
- And add some hooks into `tf.estimator.Estimator.train`

**_References:_**

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

**_References:_**

- [TensorFlow API: tf.train.exponential_decay](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay)

### Load pre-trained model

**_References:_**

- [Github tensorflow/tensorflow Issues: Estimator should be able to partially load checkpoints #10155](https://github.com/tensorflow/tensorflow/issues/10155)

<!--  -->
<br>

---

<!--  -->

# Erros & Solutions

- `TensorFlow ValueError: Cannot feed value of shape (1, 64, 64, 3) for Tensor u'Placeholder:0', which has shape '(1, ?, ?, 1)'`
  It means that you feed wrong data shape into TensorFlow placeholder.

  **_References:_**

  - [stackoverflow: TensorFlow ValueError: Cannot feed value of shape (64, 64, 3) for Tensor u'Placeholder:0', which has shape '(?, 64, 64, 3)'](https://stackoverflow.com/questions/40430186/tensorflow-valueerror-cannot-feed-value-of-shape-64-64-3-for-tensor-uplace)

- `TensorFlow: “Attempting to use uninitialized value” in variable initialization`
  It means that you run your sess without initialise Variables. May you run variable initialise and train/predict in two different session?
  **_References:_**

  - [stackoverflow: TensorFlow: “Attempting to use uninitialized value” in variable initialization](https://stackoverflow.com/questions/44624648/tensorflow-attempting-to-use-uninitialized-value-in-variable-initialization/44630421)

- When import tensorflow, occur error `ImportError: libcublas.so.9.0: cannot open shared object file: No such file or director`
  It means that maybe you use wrong `cuda` version. You need to set `export LD_LIBRARY_PATH=<your cuda install path>:$LD_LIBRARY_PATH`

  **_References:_**

  - [Github tensorflow/tensorflow Issue: ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory #15604](https://github.com/tensorflow/tensorflow/issues/15604)
  - [stackoverflow: ImportError: libcublas.so.9.0: cannot open shared object file](https://stackoverflow.com/questions/48428415/importerror-libcublas-so-9-0-cannot-open-shared-object-file)

- When do `x / 127.5 - 1` occur error `TypeError: unsupported operand type(s) for /: 'Tensor' and 'float'`
  Maybe your `x` is not `tf.float32`, you can try `x = tf.cast(x, tf.float32)`

- When using `tf.data.Dataset` occur error `Tensorflow GetNext() failed because the iterator has not been initialized`

  **_References:_**

  - [stackoverflow: Tensorflow GetNext() failed because the iterator has not been initialized](https://stackoverflow.com/questions/48443203/tensorflow-getnext-failed-because-the-iterator-has-not-been-initialized)
  - [TensorFlow API make_initializable_iterator](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#make_initializable_iterator)

<!--  -->
<br>

---

<!--  -->

# QA

## Tensorflow `tf.identity()` do for what?

When I watch `Tensorflow Models/resnet_model.py`, it use `inputs = tf.identity(inputs, 'initial_conv')`. Here is a good explanation from [Zhihu](https://zhuanlan.zhihu.com/p/32540546)

**_References:_**

- [知乎: TensorFlow 的 Graph 计算流程控制](https://zhuanlan.zhihu.com/p/32540546)

## Set learning rate decay in Tensorflow

A common function is `tf.train.exponential_decay`. It use `global_step` to calculate learning rate. TensorFlow `Optimimizer.minimize` need a args `global_step`. `global_step` will add 1 each batch. So if you want to update learning rate in epoch, you need to set `decay_step=<step> * num_batch`

**_References:_**

- [知乎: Tensorflow 中 learning rate decay 的奇技淫巧](https://zhuanlan.zhihu.com/p/32923584)
- [知乎: 使用 Tensorflow 过程中遇到过哪些坑？](https://www.zhihu.com/question/269968195)
- [stackoverflow: How to set adaptive learning rate for GradientDescentOptimizer?](https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer/33922859)
- [Blog: Tensorflow 一些常用基本概念与函数（四）](https://www.cnblogs.com/wuzhitj/p/6648641.html)

<!--  -->
<br>

---

<!--  -->

# TFLite

Begin with tensorflow >= 1.12, tflite is not in contrib.

## Convert

**Tips: Better to use SavedModel.**

### Convert SavedModel into TFLite

**_References:_**

- [TensorFlow Lite Guide: Convert the model format](https://www.tensorflow.org/lite/devguide#2_convert_the_model_format)
- [TensoFlow Lite Guide: Converter command-line examples](https://www.tensorflow.org/lite/convert/cmdline_examples#convert_a_tensorflow_graphdef_)

---

## Build

**Note:** Better build in a docker container.

1. Install Python, refer to [tensorflow build from src][tensorflow build from src].

2. Install bazel, refer to [install bazel][install bazel].

3. Run `./configure`, and set NDK and Android SDK path (**Please use ndk >= 18**).

4. Edit `tensorflow/contrib/lite/build`, add followings

   ```bash
   cc_binary(
       name = "libtensorflowLite.so",
       linkopts=[
           "-shared",
           "-Wl,-soname=libtensorflowLite.so",
       ],
       linkshared = 1,
       copts = tflite_copts(),
       deps = [
           ":framework",
           "//tensorflow/contrib/lite/kernels:builtin_ops",
       ],
   )
   ```

5. Build

   - Android

     ```bash
     bazel build //tensorflow/contrib/lite:libtensorflowLite.so --crosstool_top=//external:android/crosstool --cpu=armeabi-v7a --host_crosstool_top=@bazel_tools//tools/cpp:toolchain --cxxopt="-std=c++11"
     ```

   - Linux

     ```bash
     bazel build //tensorflow/contrib/lite:libtensorflowLite.so --cxxopt="-std=c++11"
     ```

   step 3 and 4 refer to [stackoverflow][stackoverflow].

[tensorflow build from src]: https://www.tensorflow.org/install/source?hl=zh-cn#setup_for_linux_and_macos
[install bazel]: https://docs.bazel.build/versions/master/install.html
[stackoverflow]: https://stackoverflow.com/questions/49834875/problems-with-using-tensorflow-lite-c-api-in-android-studio-project

**_Referneces:_**

- [stackoverflow: Problems with using tensorflow lite C++ API in Android Studio Project](https://stackoverflow.com/questions/49834875/problems-with-using-tensorflow-lite-c-api-in-android-studio-project)
- [TensorFlow doc: 从源码构建](https://www.tensorflow.org/install/source?hl=zh-cn)
- [Medium: Bazel build C++ Tensorflow Lite library for Android (without JNI)](https://medium.com/@punpun/bazel-build-c-tensorflow-lite-library-for-android-without-jni-f92b87aa9610)
- [Blog: Tensorflow Lite 编译](https://fucknmb.com/2017/11/17/Tensorflow-Lite%E7%BC%96%E8%AF%91/)
- [Blog: Android App With Tflite C++ Api](http://www.sanjaynair.one/Android-App-With-Tflite-C++-API/)

### [Android] Problems during buiding

#### `WARNING: The following rc files are no longer being read, please transfer their contents or import their path into one of the standard rc files:/home/gopi/tensorflow/tools/bazel.rc`

**_Solution:_**

- [Github tensorflow/tensorflow: Build from source -> build the pip package -> GPU support -> bazel build -> ERROR: Config value cuda is not defined in any .rc file #23401](https://github.com/tensorflow/tensorflow/issues/23401#issuecomment-435827786)

---

## Using

- Include:
  - `flatbuffers`
  - `tensorflow/lite`
- Link
  - `libtensorflowLite.so`

Infer using TFLite C++ api, here is a example

```c++
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"

tflite::FlatBufferModel model(path_to_model);
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Resize input tensors, if desired.
//

interpreter->AllocateTensors();
float* input = interpreter->typed_input_tensor<float>(0);

// Fill `input` e.g. feed an image
float* input interpreter->typed_input_tensor<float>(0);
for (unsigned int i = 0; i < 128 * 128 * 3; ++i)
{
    input[i] = img_data[i];
}

interpreter->Invoke();
float* output = interpreter->typed_output_tensor<float>(0);
```

**_References:_**

- [TensorFlow doc guide: TensorFlow Lite APIs](https://www.tensorflow.org/lite/apis#c)

### [Android] Problems during compile

#### `undefined reference to 'tflite::InterpreterBuilder::operator()`

**_Solution:_**

You must use **ndk(>=18)** to build the TFLite library and set `Android NDK location` as ndk path.

**Important note:** ndk18 has removed `libgnustl` which is stl library for C++, replaced with `c++_static/c++_shared`, _refer to [NDK Revision History][ndk revision history]_. So all your third party C++ libs should built with `c++_static/c++_shared` not with `libgnustl`.

And you need to set following in the android project app gradle:

```bash
externalNativeBuild {
    cmake {
        cppFlags "-std=c++11 -fopenmp -O3 -llog -landroid"
        abiFilters "armeabi-v7a"
        arguments '-DAPP_STL=c++_static
    }
}
```

`arguments '-DAPP_STL=c++_static` refer to [NDK C++ support][ndk c++ support]

[ndk c++ support]: https://developer.android.com/ndk/guides/cpp-support?hl=zh-cn#header
[ndk revision history]: https://developer.android.com/ndk/downloads/revision_history

<br>

---

old version

<br>

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

<br>

---

<br>

# Functions

## tf

### tf.Session()

### tf.Graph()

#### tf.InteractiveSession()

A Tensorflow `Session` for use in interactive contexts, such asashell.

```python
sess = tf.InteractiveSession()
a = tf.constant(5)
b = tf.constant(6)
c = a * b
print(c.eval())
```

### Create constant tensor\*\*

- `tf.zeros(shape, dtype=tf.float32, name=None)`
- `tf.zeros_like(input_tensor, dtype=None, name=None, optimize=True`
- `tf.fill(dims, value, name=None)`: Create a tensor filled with ascalar value

### Random

- `tf.set_random_seed(seed)`: Set the graph-level random seed. Youcan use it after your build a graph and before run a session.

- `tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32,seed=None, name=None)`

- `tf.truncated_normal(shape, mean=0.0, stddev=1.0,dtype=tf.float32, seed=None, name=None)`

- `tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32 seed=None, name=None)`

- `tf.random_shuffle(value, seed=None, name=None)`

### Variable

**All variable in Tensorflow need to be initialized!** It means that when you run a session, you need to init the variable before doing other ops. A good way is to use `tf.global_variables_initializer()`

```python
# Create two variables.
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),name="weights")
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

### Placeholder

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
```

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

  **_References_**

  - [Tensorflow API: Constants, Sequences, and Random Values](https://www.tensorflow.org/api_guides/python/constant_op#Random_Tensors)
  - [Tensorflow Programmers's Guide: Variables: Creation, Initialization, Saving, and Loading](https://www.tensorflow.org/versions/r1.0/programmers_guide/variables)

### tf.slice

Extracts a slice from a tensor.

### tf.summary and Tensorboard

Start Tensorboard

```shell
tensorboard --logdir=<tensorflow run log path> [--port]
```

A example shows how to write `tf.summary`: [Blog: Tensorflow 学习笔记——Summary 用法](https://www.cnblogs.com/lyc-seu/p/8647792.html)

<br>

---

<br>

# Websites

- [Stanford CS 20: Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/syllabus.html)

# tf.nn, tf.layers, tf.contrib 区别

**_References:_**

- [小鹏的专栏: tf API 研读 1：tf.nn，tf.layers， tf.contrib 概述](https://cloud.tencent.com/developer/article/1016697)

<br>

---

<br>