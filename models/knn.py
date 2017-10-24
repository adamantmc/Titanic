from sklearn.neighbors import KNeighborsClassifier

from models.titanic_model import TitanicModel

class KNeighboursModel(TitanicModel):

    def __init__(self, train_x, train_y, k):
        super(KNeighboursModel, self).__init__()
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.model.predict(test_x)