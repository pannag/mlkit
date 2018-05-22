#!/usr/bin/env python

"""
Creates a simple model with unit weights and zero bias.
The input to the model is a 64 dimensional vector.
The output is the logits (without the softmax) and thus amounts
to the sum of the input values given our model.

The model is frozen and converted to tflite and saved to disk.
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

# Write graph and convert to tflite
tf.reset_default_graph()

input = tf.placeholder(name="input", dtype=tf.float32, shape=[1, 64])
fc = layers.fully_connected(
  input,
  num_outputs=1,
  weights_initializer=tf.ones_initializer,
  biases_initializer=tf.zeros_initializer,
  activation_fn=None)
out = tf.identity(fc, name="out")

with tf.Session() as sess:
  init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  sess.run(init)
  frozen_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, sess.graph_def, output_node_names=["out"])

with open("frozen_graph.pbtxt", "w") as fp:
  fp.write(str(frozen_graph_def))
# print the nodes in frozen_graph_def
for node in frozen_graph_def.node:
  print(node.name)

tflite_model = tf.contrib.lite.toco_convert(frozen_graph_def, [input], [out])

open("converted_model.tflite", "wb").write(tflite_model)
