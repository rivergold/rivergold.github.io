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
...

In computation graph:
- Nodes is: operators, variables or constants
-Edges: tensors

## Why graphs
1. Save computation. Only run subgraphs that lead to the values you want to fetch.
2. Break computation into small, differential pieces to facilitate auto-differentiation.
3. Facilitate distributed computation, spread the work across multiple CPUs, GPUs, TPUs or other devices.

# Constants
```
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
```sh
$ tensorboard --logdir=<tensorflow run log path> [--port]
```






# Websites
- [Stanford CS 20: Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/syllabus.html)