from models.titanic_model import TitanicModel
from sklearn.ensemble import AdaBoostClassifier

class AdaBoostModel(TitanicModel):

    def __init__(self, train_x, train_y):
        super(AdaBoostModel, self).__init__()

        self.model = AdaBoostClassifier(n_estimators=50)
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.model.predict(test_x)