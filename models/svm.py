from sklearn.svm import SVC
from models.titanic_model import TitanicModel

class SVMModel(TitanicModel):

    def __init__(self, train_x, train_y, kernel):
        self.model = SVC(kernel=kernel)
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.model.predict(test_x)
