"""
Example WeightNorm eager execution
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.layers import WeightNorm

tf.enable_eager_execution()

# Parameters
learning_rate = 0.001
momentum = 0.9
num_epochs = 10
batch_size = 256
n_classes = 10


def weightnorm_net():
    model = tf.keras.Sequential()
    model.add(WeightNorm(tf.layers.Conv2D(6, 5, activation='relu'),
                         input_shape=(32, 32, 3)))

    model.add(tf.layers.MaxPooling2D(2, 2))

    model.add(WeightNorm(tf.layers.Conv2D(16, 5, activation='relu')))
    model.add(tf.layers.MaxPooling2D(2, 2))

    model.add(tf.layers.Flatten())
    model.add(WeightNorm(tf.layers.Dense(120, activation='relu')))
    model.add(WeightNorm(tf.layers.Dense(84, activation='relu')))
    model.add(WeightNorm(tf.layers.Dense(n_classes)))

    return model


def weightnorm_keras_net():
    model = tf.keras.Sequential()
    model.add(WeightNorm(tf.keras.layers.Conv2D(6, 5, activation='relu'),
                         input_shape=(32, 32, 3)))

    model.add(tf.keras.layers.MaxPooling2D(2, 2))

    model.add(WeightNorm(tf.keras.layers.Conv2D(16, 5, activation='relu')))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))

    model.add(tf.keras.layers.Flatten())
    model.add(WeightNorm(tf.keras.layers.Dense(120, activation='relu')))
    model.add(WeightNorm(tf.keras.layers.Dense(84, activation='relu')))
    model.add(WeightNorm(tf.keras.layers.Dense(n_classes)))

    return model


def regular_net():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(6, 5, activation='relu'))

    model.add(tf.layers.MaxPooling2D(2, 2))

    model.add(tf.keras.layers.Conv2D(16, 5, activation='relu'))
    model.add(tf.layers.MaxPooling2D(2, 2))

    model.add(tf.layers.Flatten())
    model.add(tf.keras.layers.Dense(120, activation='relu'))
    model.add(tf.keras.layers.Dense(84, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    return model


def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    labels = tf.cast(labels, tf.int64)
    return tf.reduce_sum(
        tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def loss_function(model, x, y):
    y_ = model(x)
    return tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(logits=y_, labels=y))


if __name__ == "__main__":
    (train_x, train_y), (test_x, test_y) = load_data()
    train_x, test_x = train_x.astype(np.float32), test_x.astype(np.float32)
    train_y, test_y = train_y.astype(np.int), test_y.astype(np.int)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.shuffle(train_x.shape[0])
    train_dataset = train_dataset.repeat(num_epochs)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.batch(batch_size)

    model = weightnorm_keras_net()

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=momentum)

    for (x, y) in tfe.Iterator(train_dataset):
        x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),
                      x, dtype=tf.float32)

        grads = tfe.implicit_gradients(loss_function)(model, x, y)
        optimizer.apply_gradients(grads)

    accuracy = tfe.metrics.Accuracy('accuracy')
    for (x, y) in tfe.Iterator(test_dataset):
        x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame),
                      x, dtype=tf.float32)

        y = tf.reshape(y, [-1])
        logits = model(x)
        accuracy(tf.argmax(logits, axis=1, output_type=tf.int64),
                 tf.cast(y, tf.int64))

    print('Test set:Accuracy: %4f%%\n' % (100 * accuracy.result()))
