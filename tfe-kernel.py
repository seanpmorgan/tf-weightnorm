import tensorflow as tf

tf.enable_eager_execution()

if __name__ == "__main__":
    input_size = 3
    hidden_size = 3
    batch = tf.zeros([1, input_size])

    layer = tf.layers.Dense(units=hidden_size,
                            kernel_initializer=tf.constant_initializer(5))
    layer(batch)
    # print(layer.kernel)
    print(layer.trainable_variables)
    print(layer.non_trainable_variables)
