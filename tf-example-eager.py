"""
Example WeightNorm eager execution

Collab:
https://colab.research.google.com/drive/1nBQSAA78oUBmi9fhnHJ_zWhHq2NXjwIc
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data
from normalization import WeightNorm
from matplotlib import pyplot as plt

tf.enable_eager_execution()
tfe = tf.contrib.eager


class WnModel(tf.keras.Model):
    def __init__(self):
        super(WnModel, self).__init__()
        self.maxpool = tf.layers.MaxPooling2D(2, 2)

        self.conv1 = WeightNorm(tf.layers.Conv2D(6, 5, activation='relu'),
                                input_shape=(32, 32, 3))

        self.conv2 = WeightNorm(tf.layers.Conv2D(16, 5, activation='relu'))

        self.flatten = tf.layers.Flatten()
        self.dense1 = WeightNorm(tf.layers.Dense(120, activation='relu'))
        self.dense2 = WeightNorm(tf.layers.Dense(84, activation='relu'))
        self.dense3 = WeightNorm(tf.layers.Dense(n_classes))

    def call(self, input):
        x = self.maxpool(self.conv1(input))
        x = self.maxpool(self.conv2(x))
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class RegularModel(tf.keras.Model):
    def __init__(self):
        super(RegularModel, self).__init__()
        self.maxpool = tf.layers.MaxPooling2D(2, 2)

        self.conv1 = tf.layers.Conv2D(6, 5, activation='relu')
        self.conv2 = tf.layers.Conv2D(16, 5, activation='relu')

        self.flatten = tf.layers.Flatten()
        self.dense1 = tf.layers.Dense(120, activation='relu')
        self.dense2 = tf.layers.Dense(84, activation='relu')
        self.dense3 = tf.layers.Dense(n_classes)

    def call(self, input):
        x = self.maxpool(self.conv1(input))
        x = self.maxpool(self.conv2(x))
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    labels = tf.cast(labels, tf.int64)
    return tf.reduce_sum(
        tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def loss_function(model, x, y):
    y_ = model(x)

    return tf.reduce_mean(
        tf.losses.sparse_softmax_cross_entropy(logits=y_, labels=y))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_function(model, inputs, targets)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train_and_eval(model, print_grads=False):

    accuracy = tfe.metrics.Accuracy('accuracy')
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=momentum,)

    # Train Model
    train_losses = []
    step = 0

    for (x, y) in tfe.Iterator(train_dataset):
        step += 1
        x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x, dtype=tf.float32)

        loss_value, grads = grad(model, x, y)
        if print_grads:
            for update in list(zip(grads, model.variables)):
                    print(update)
                    print('\n\n')

        optimizer.apply_gradients(zip(grads, model.variables), global_step)

        if step % 10 == (10 - 1):
            train_losses.append(loss_value)

    # Test Model
    for (x, y) in tfe.Iterator(test_dataset):
        x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x, dtype=tf.float32)
        y = tf.reshape(y, [-1])

        logits = model(x)
        accuracy(tf.argmax(logits, axis=1, output_type=tf.int64),
                 tf.cast(y, tf.int64))

    return np.asarray(train_losses), (100 * accuracy.result())


if __name__ == "__main__":

    # Parameters
    learning_rate = 0.001
    momentum = 0.9
    num_epochs = 1
    batch_size = 256
    n_classes = 10

    (train_x, train_y), (test_x, test_y) = load_data()
    train_x, test_x = train_x.astype(np.float32), test_x.astype(np.float32)
    train_y, test_y = train_y.astype(np.int), test_y.astype(np.int)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.shuffle(train_x.shape[0])
    train_dataset = train_dataset.repeat(num_epochs)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
    test_dataset = test_dataset.batch(batch_size)

    with tf.device("/gpu:0"):
        weightnorm_keras_model = WnModel()
        regular_model = RegularModel()

    wn_keras_loss, wn_keras_accuracy = train_and_eval(weightnorm_keras_model)
    regular_loss, regular_accuracy = train_and_eval(regular_model)

    size = wn_keras_loss.shape[0]
    plt.plot(np.linspace(0, size,  size), wn_keras_loss,
             color='green', label='keras-weightnorm')

    plt.plot(np.linspace(0, size, size), regular_loss,
             color='blue', label='regular parameterization')

    plt.legend()
    plt.show()

    print('Regular accuracy: {0}'.format(regular_accuracy))
    print('Wn Keras accuracy: {0}'.format(wn_keras_accuracy))
