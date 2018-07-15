import tensorflow as tf
from tensorflow.contrib.layers import weight_normalization, WeightNormalization


# def object_based():
#     batch = tf.zeros([1, input_size])
#
#     dense_args = {"units": hidden_size,
#                   "kernel_initializer": tf.constant_initializer(5),
#                   "activation": None}
#
#     layer_1 = WeightNormalization(tf.layers.Dense, **dense_args)
#     layer_1.apply(batch)
#
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
#         print(layer_1.layer.trainable_variables)
#         print(layer_1.variables)
#         print(sess.run(layer_1.layer.kernel))
#
#
# def fn_based():
#     batch = tf.zeros([1, input_size])
#
#     layer_1 = tf.layers.dense(batch, units=hidden_size,
#                               kernel_initializer=tf.constant_initializer(5))
#
#     init = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init)
#         print(sess.run(layer_1.trainable_variables))


if __name__ == "__main__":
    input_size = 3
    hidden_size = 3

    # object_based()
    # fn_based()
