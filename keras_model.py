from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class KerasModel(BaseEstimator, ClassifierMixin):

    """
    Class that implements a neural network in Keras, implementing a
    scikit estimator for use in GridSearchCV.
    """

    def __init__(self, learning_rate=0.001,
                 batch_size=64, epochs=20,
                 layers=[256],
                 activation_function="relu", loss_function="binary_crossentropy",
                 verbose=False):
        super(KerasModel, self).__init__()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.layers = layers
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.verbose = verbose

    def fit(self, train_x, train_y):
        self.network = Sequential()

        self.network.add(Dense(self.layers[0], input_shape=(len(train_x[0]),), activation=self.activation_function))
        if len(self.layers) > 1:
            for layers in self.layers[1:]:
                self.network.add(Dense(layers, activation=self.activation_function))

        self.network.add(Dense(1, activation=self.activation_function))

        optimizer = Adam(lr=self.learning_rate)

        self.network.compile(optimizer=optimizer,
                             loss=self.loss_function,
                             metrics=['accuracy'])

        self.network.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, test_x):
        predictions = [self._prob_to_label(pred[0]) for pred in self.network.predict(test_x)]
        return predictions

    def _prob_to_label(self, val):
        if val > 0.5:
            return 1
        else:
            return 0

