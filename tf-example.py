import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.contrib.layers import weight_normalization


def regular_net(x, n_classes):
    with tf.variable_scope('Regular'):
        net = tf.layers.conv2d(x, 6, 5)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = tf.layers.conv2d(net, 16, 5)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, 2, 2)

        net = tf.layers.flatten(net)

        net = tf.layers.dense(net, 120)
        net = tf.nn.relu(net)

        net = tf.layers.dense(net, 84)
        net = tf.nn.relu(net)

        net = tf.layers.dense(net, n_classes)

        return net


def train(x, y, num_epochs, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.shuffle(x.shape[0])
    train_dataset = train_dataset.repeat(num_epochs)
    train_dataset = train_dataset.batch(batch_size)
    iterator = train_dataset.make_initializable_iterator()

    inputs, labels = iterator.get_next()
    inputs = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), inputs, dtype=tf.float32)

    inputs = tf.Print(inputs, [inputs])

    logits = regular_net(inputs, 10)

    logits = tf.Print(logits, [logits])
    labels = tf.Print(labels, [labels])

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    train_op = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()

    step = 0
    running_loss = 0
    running_loss_array = []

    with tf.Session() as sess:
        sess.run(init)
        sess.run(iterator.initializer)

        while True:
            try:
                _, loss = sess.run([train_op, loss_op])
                step += 1
                running_loss += (loss / batch_size)
                if step % 32 == (32 - 1):
                    print(step, running_loss)
                    running_loss_array.append(running_loss)
                    running_loss = 0.0

            except tf.errors.OutOfRangeError:
                return running_loss_array


if __name__ == "__main__":
    learning_rate = 0.001
    momentum = 0.9
    num_epochs = 2
    batch_size = 128
    n_classes = 10
    (train_x, train_y), (test_x, test_y) = load_data()

    train_x = train_x.astype(float)
    train_y = train_y.astype(float)

    regular_loss = train(train_x, train_y, num_epochs, batch_size)
    regular_loss = np.asarray(regular_loss)

    num_data = regular_loss.shape[0]
    plt.plot(np.linspace(0, num_data, num_data), regular_loss,
             color='blue', label='regular parameterization')
    plt.legend()
    plt.show()
