from models.titanic_model import TitanicModel
from xgboost import XGBClassifier
import numpy as np

class XGBoostModel(TitanicModel):

    def __init__(self, train_x, train_y):
        self.model = XGBClassifier(n_estimators=50)
        self.model.fit(np.asarray(train_x), np.asarray(train_y))

    def predict(self, test_x):
        return self.model.predict(np.asarray(test_x))