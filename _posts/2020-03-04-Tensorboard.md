---
title: "Tensorboard"
last_modified_at: 2020-03-04
categories:
  - Memo
tags:
  - Pytorch
  - Tool
  - Memo
---

General usage of Tensorboard.

## Install

```shell
pip install tensorboard
```

PyTorch also supports Tensorboad.

## :fallen_leaf:Summary

A summary is basically a special kind of TensorFlow Op. It will take in a regular tensor from your graph and then it will output protocol buffers that we can write to disk.

There are a couple of different kinds of summaries:

- `tf.summary.scalar`
- `tf.summary.image`
- `tf.summary.audio`
- `tf.summary.histogram`: Let you look at the shape of distribution of different variables, very nice to be attached to model variables, like your weights.
- `tf.summary.tensor`: it can write out any kind of value

### How Summaries Work

1. Summary Op returns protocol buffers
2. tf.summary.FileWriter writes them to disk

We run each summary Op, It gives us the protocol buffers and then we can pass them to our File Writer to get them to disk.

There maybe have many summaries in the graph, so we have `tf.summary.merge_all()`. It will merge all summaries that creates a single target. We run it and we get every summary in the whole graph.

## :fallen_leaf:Embedding Visualizer

Let you take high dimensional data and then projet it down into three dimensions.

### What is Embedding

When we take our input data set and we map it through the neural network to see the final layer. The embedding is actually the learned representation of how our neural network is processing the information.

**_Ref_** :thumbsup::thumbsup::thumbsup:[TensorFlow: Hands-on TensorBoard (TF Dev Summit '17)](https://www.youtube.com/watch?v=eBbEDRsCmv4)
