import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from models.titanic_model import TitanicModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import subprocess

class DecisionTreeModel(TitanicModel):

    def __init__(self, train_x, train_y, criterion):
        super(DecisionTreeModel, self).__init__()

        self.model = DecisionTreeClassifier(criterion=criterion)
        self.model.fit(train_x, train_y)

    def visualize_png(self, name):
        export_graphviz(self.model, "tree.dot")
        subprocess.call(["dot", "-Tpng", "tree.dot", "-o {}_tree.png".format(name)])
        subprocess.call(["rm", "tree.dot"])

    def predict(self, test_x):
        return self.model.predict(test_x)
