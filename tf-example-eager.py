import tensorflow as tf
import tensorflow.contrib.eager as tfe
import mnist_data as mnist_dataset
from tensorflow.contrib.layers import WeightNormalization

tf.enable_eager_execution()

# Parameters
learning_rate = 0.001
num_epochs = 1
batch_size = 100
display_step = 1

# Network Parameters
n_input = 784
n_hidden_1 = 500
n_classes = 10
dense_args = {'units': n_hidden_1, 'activation': tf.nn.relu}


class Net(tfe.Network):
    def __init__(self):
        super(Net, self).__init__()
        # self.fc1 = self.track_layer(tf.layers.Dense(units=n_hidden_1, activation=tf.nn.relu))
        self.fc1 = self.track_layer(WeightNormalization(tf.layers.Dense, **dense_args))
        self.fc2 = self.track_layer(tf.layers.Dense(units=n_classes))

    def call(self, input):
        result = self.fc1(input)
        result = self.fc2(result)
        return result


def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    labels = tf.cast(labels, tf.int64)
    batch_size = int(logits.shape[0])
    return tf.reduce_sum(
        tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def loss_function(model, x, y):
    y_ = model(x)
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y_, labels=y))


if __name__ == "__main__":
    # Load the datasets
    train_ds = mnist_dataset.train('data/raw').shuffle(80000).batch(batch_size)
    test_ds = mnist_dataset.test('data/raw').batch(batch_size)

    model = Net()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    for (x, y) in tfe.Iterator(train_ds):
        grads = tfe.implicit_gradients(loss_function)(model, x, y)
        optimizer.apply_gradients(grads)

        # Alternatively
        # optimizer.minimize(lambda: loss_function(model, x, y))

    accuracy = tfe.metrics.Accuracy('accuracy')
    for (x, y) in tfe.Iterator(test_ds):
        logits = model(x)
        accuracy(tf.argmax(logits, axis=1, output_type=tf.int64),
                 tf.cast(y, tf.int64))

    print('Test set:Accuracy: %4f%%\n' % (100 * accuracy.result()))
