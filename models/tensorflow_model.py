import tensorflow as tf
from models.titanic_model import TitanicModel

class TensorFlowModel(TitanicModel):

    def __init__(self, train_x, train_y, learning_rate, batch_size, epochs, verbose=False):
        super(TensorFlowModel, self).__init__()

        self.x = tf.placeholder(tf.float32, shape=[None, len(train_x[0])])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])

        layer_1 = tf.layers.dense(inputs=self.x, units=256, activation=tf.nn.relu)
        layer_2 = tf.layers.dense(inputs=layer_1, units=2)

        self.logits = layer_2

        loss_func = tf.losses.softmax_cross_entropy(logits=self.logits, onehot_labels=self.y)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_func)

        correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        init = tf.global_variables_initializer()

        self.session = tf.Session()
        self.session.run(init)

        for epoch in range(epochs):
            start, end = 0, batch_size

            while end <= len(train_x):
                x_batch = train_x[start:end]
                y_batch = train_y[start:end]

                feed_dict = {self.x: x_batch, self.y: y_batch}
                self.session.run(train_op, feed_dict=feed_dict)

                start += batch_size
                end += batch_size

                loss, acc = self.session.run([loss_func, accuracy], feed_dict=feed_dict)

                if verbose:
                    print("Epoch: {} Batch: {}-{} Loss: {} Accuracy: {}".format(epoch + 1, start, end, loss, acc))

    def predict(self, test_x):
        feed_dict = {self.x:test_x}
        predictions = [self.argmax(pred) for pred in self.session.run(self.logits, feed_dict=feed_dict)]

        return predictions
