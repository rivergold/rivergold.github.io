# Erros & Solutions

## Error from `tf.keras.losses.mean_absolute_error` when calcualte `L1` loss

Key Code

```python
# loss
loss = tf.keras.losses.mean_absolute_error(labels, y)

# Compute evaluation metrics.
mea = tf.metrics.mean_absolute_error(labels=labels,
                                              predictions=y,
                                              name='mae_op')
metrics = {'mea': mea}
tf.summary.scalar('mea', mea[1])

if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

# Create training op.
assert mode == tf.estimator.ModeKeys.TRAIN

optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
```

```
Traceback (most recent call last):
  File "train.py", line 66, in <module>
    estimator.train(input_fn=input_fn)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 363, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 843, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 856, in _train_model_default
    features, labels, model_fn_lib.ModeKeys.TRAIN, self.config)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 831, in _call_model_fn
    model_fn_results = self._model_fn(features=features, **kwargs)
  File "/home/rivergold/Desktop/noise2noise/model/srgan.py", line 43, in srgan
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/model_fn.py", line 188, in __new__
    loss = array_ops.reshape(loss, [])
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/gen_array_ops.py", line 6113, in reshape
    "Reshape", tensor=tensor, shape=shape, name=name)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 787, in _apply_op_helper
    op_def=op_def)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 3392, in create_op
    op_def=op_def)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1734, in __init__
    control_input_ops)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1570, in _create_c_op
    raise ValueError(str(e))
ValueError: Dimension size must be evenly divisible by 15876 but is 1 for 'Reshape' (op: 'Reshape') with input shapes: [?,126,126], [0].
```

When use `tf.keras.losses.mean_absolute_error` calculate two image `L1` loss, occur this error.

Question: How to calculate L1 loss in Tensorflow.

## Loss `NaN`

Key code is

```python
def srgan(features, labels, mode, params):
    def _residual_block(features):
        x = tf.layers.conv2d(features, 64, (3, 3), padding='same', kernel_initializer=tf.truncated_normal_initializer())
        # x = tf.layers.batch_normalization(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

        x = tf.Print(x, [x])

        x = tf.layers.conv2d(x, 64, (3, 3), padding='same', kernel_initializer=tf.truncated_normal_initializer())
        # x = tf.layers.batch_normalization(x)
        m = tf.keras.layers.Add()([x, features])
        return m

    x = tf.layers.conv2d(features, 64, (3, 3), padding='same', kernel_initializer=tf.truncated_normal_initializer())
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    x = tf.Print(x, [x])

    x0 = x

    for i in range(16):
        x = _residual_block(x)
    x = tf.layers.conv2d(x, 64, (3, 3), padding='same', kernel_initializer=tf.truncated_normal_initializer())
    # x = tf.layers.batch_normalization(x)
    x = tf.keras.layers.Add()([x, x0])
    y = tf.layers.conv2d(x, 3, (3, 3), padding='same', kernel_initializer=tf.truncated_normal_initializer())

    y = tf.Print(y, [y])

    # loss
    loss = tf.losses.mean_squared_error(labels, y)
```

```
ERROR:tensorflow:Model diverged with loss = NaN.
Traceback (most recent call last):
  File "train.py", line 66, in <module>
    estimator.train(input_fn=input_fn)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 363, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 843, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 859, in _train_model_default
    saving_listeners)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 1059, in _train_with_estimator_spec
    _, loss = mon_sess.run([estimator_spec.train_op, estimator_spec.loss])
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 567, in run
    run_metadata=run_metadata)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1043, in run
    run_metadata=run_metadata)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1134, in run
    raise six.reraise(*original_exc_info)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/six.py", line 693, in reraise
    raise value
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1119, in run
    return self._sess.run(*args, **kwargs)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/training/monitored_session.py", line 1199, in run
    run_metadata=run_metadata))
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/training/basic_session_run_hooks.py", line 623, in after_run
    raise NanLossDuringTrainingError
tensorflow.python.training.basic_session_run_hooks.NanLossDuringTrainingError: NaN loss during training.
```

The loss is caused by `tf.truncated_normal_initializer()`. When I change it into `tf.keras.initializers.he_normal()`, the loss is ok.

## `tf.layers.conv2d` cannot know the input shape.

Using `tf.py_func` process tensor and put the return into model, the model layer cannot recognize the shape, and occur error like,

```
WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmpkzwvdjlq
WARNING:tensorflow:Estimator's model_fn (<function srgan at 0x7f9bfa8067b8>) includes params argument, but params are not passed to Estimator.
Traceback (most recent call last):
  File "train.py", line 66, in <module>
    estimator.train(input_fn=input_fn)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 363, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 843, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 856, in _train_model_default
    features, labels, model_fn_lib.ModeKeys.TRAIN, self.config)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 831, in _call_model_fn
    model_fn_results = self._model_fn(features=features, **kwargs)
  File "/home/rivergold/Desktop/noise2noise/model/srgan.py", line 16, in srgan
    x = tf.layers.conv2d(features, 64, (3, 3), padding='same', kernel_initializer=tf.truncated_normal_initializer())
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/layers/convolutional.py", line 621, in conv2d
    return layer.apply(inputs)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/layers/base.py", line 828, in apply
    return self.__call__(inputs, *args, **kwargs)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/layers/base.py", line 691, in __call__
    self._assert_input_compatibility(inputs)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/layers/base.py", line 1174, in _assert_input_compatibility
    self.name + ' is incompatible with the layer: '
ValueError: Input 0 of layer conv2d_1 is incompatible with the layer: its rank is undefined, but the layer requires a defined rank.
```

**_References:_**

- [Github tensorflow/tensorflow Issues: Dataset API does not pass dimensionality information for its output tensor #17059](https://github.com/tensorflow/tensorflow/issues/17059)
- [Github tensorflow/tensorflow Issues: Feature Request: Setting the shape of a tf.data.Dataset if it cannot be inferred #16052](https://github.com/tensorflow/tensorflow/issues/16052)
- [Github tensorflow/tensorflow Issues: add assert_element_shape method for tf.contrib.data #17480](https://github.com/tensorflow/tensorflow/pull/17480)

**Final Solution:**
`y.set_shape(inp.get_shape())`

**_Referneces:_**

- [stackoverflow: Output from TensorFlow `py_func` has unknown rank/shape](https://stackoverflow.com/questions/42590431/output-from-tensorflow-py-func-has-unknown-rank-shape)

## Froget to cast raw image data into `tf.float32`, occur error `TypeError: Expected uint8, got 0.0 of type 'float' instead.`

Error is,

```
WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmpkkthxfdh
WARNING:tensorflow:Estimator's model_fn (<function srgan at 0x7f31dcc867b8>) includes params argument, but params are not passed to Estimator.
Traceback (most recent call last):
  File "train.py", line 66, in <module>
    estimator.train(input_fn=input_fn)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 363, in train
    loss = self._train_model(input_fn, hooks, saving_listeners)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 843, in _train_model
    return self._train_model_default(input_fn, hooks, saving_listeners)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 856, in _train_model_default
    features, labels, model_fn_lib.ModeKeys.TRAIN, self.config)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/estimator/estimator.py", line 831, in _call_model_fn
    model_fn_results = self._model_fn(features=features, **kwargs)
  File "/home/rivergold/Desktop/noise2noise/model/srgan.py", line 16, in srgan
    x = tf.layers.conv2d(features, 64, (3, 3), padding='same', kernel_initializer=tf.truncated_normal_initializer())
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/layers/convolutional.py", line 621, in conv2d
    return layer.apply(inputs)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/layers/base.py", line 828, in apply
    return self.__call__(inputs, *args, **kwargs)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/layers/base.py", line 699, in __call__
    self.build(input_shapes)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/layers/convolutional.py", line 144, in build
    dtype=self.dtype)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/layers/base.py", line 546, in add_variable
    partitioner=partitioner)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/training/checkpointable.py", line 436, in _add_variable_with_custom_getter
    **kwargs_for_getter)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/variable_scope.py", line 1317, in get_variable
    constraint=constraint)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/variable_scope.py", line 1079, in get_variable
    constraint=constraint)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/variable_scope.py", line 425, in get_variable
    constraint=constraint)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/variable_scope.py", line 394, in _true_getter
    use_resource=use_resource, constraint=constraint)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/variable_scope.py", line 786, in _get_single_variable
    use_resource=use_resource)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/variable_scope.py", line 2220, in variable
    use_resource=use_resource)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/variable_scope.py", line 2210, in <lambda>
    previous_getter = lambda **kwargs: default_variable_creator(None, **kwargs)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/variable_scope.py", line 2193, in default_variable_creator
    constraint=constraint)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/variables.py", line 235, in __init__
    constraint=constraint)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/variables.py", line 343, in _init_from_args
    initial_value(), name="initial_value", dtype=dtype)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/variable_scope.py", line 770, in <lambda>
    shape.as_list(), dtype=dtype, partition_info=partition_info)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/init_ops.py", line 332, in __call__
    shape, self.mean, self.stddev, dtype, seed=self.seed)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/ops/random_ops.py", line 170, in truncated_normal
    mean_tensor = ops.convert_to_tensor(mean, dtype=dtype, name="mean")
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1014, in convert_to_tensor
    as_ref=False)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1104, in internal_convert_to_tensor
    ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/framework/constant_op.py", line 235, in _constant_tensor_conversion_function
    return constant(v, dtype=dtype, name=name)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/framework/constant_op.py", line 214, in constant
    value, dtype=dtype, shape=shape, verify_shape=verify_shape))
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/framework/tensor_util.py", line 432, in make_tensor_proto
    _AssertCompatible(values, dtype)
  File "/home/rivergold/software/anaconda/lib/python3.6/site-packages/tensorflow/python/framework/tensor_util.py", line 343, in _AssertCompatible
    (dtype.name, repr(mismatch), type(mismatch).__name__))
TypeError: Expected uint8, got 0.0 of type 'float' instead.
```

**Solution:** `img_cropped = tf.cast(img_cropped, dtype=tf.float32)`
