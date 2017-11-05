from models.titanic_model import TitanicModel
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(TitanicModel):

    def __init__(self, train_x, train_y):
        super(LogisticRegressionModel, self).__init__()

        self.model = LogisticRegression()
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.model.predict(test_x)