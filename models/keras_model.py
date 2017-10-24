from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from models.titanic_model import TitanicModel

class KerasModel(TitanicModel):

    def __init__(self, train_x, train_y, learning_rate, batch_size, epochs, verbose=False):
        super(KerasModel, self).__init__()

        self.train_x = train_x
        self.train_y = train_y

        self.network = Sequential()

        self.network.add(Dense(256, input_shape=(len(train_x[0]),), activation="relu"))
        self.network.add(Dense(2, activation="softmax"))

        optimizer = Adam(lr=learning_rate)

        self.network.compile(optimizer=optimizer,
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])

        self.network.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)

        if verbose:
            print("")

    def predict(self, test_x):
        predictions = [self.argmax(pred) for pred in self.network.predict(test_x)]

        return predictions