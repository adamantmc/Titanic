from sklearn.ensemble import RandomForestClassifier
from models.titanic_model import TitanicModel

class DecisionTreeEnsemble(TitanicModel):

    def __init__(self, train_x, train_y, criterion):
        super(DecisionTreeEnsemble, self).__init__()

        self.model = RandomForestClassifier(n_estimators=1000, criterion=criterion)
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.model.predict(test_x)