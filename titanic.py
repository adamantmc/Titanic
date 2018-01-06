import csv
import pandas
import operator
import numpy as np

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import preprocessing, metrics

def one_hot(values, classes):
    vectors = np.zeros(shape=(len(values), classes))

    i = 0
    for value in values:
        vectors[i][value] = 1
        i += 1

    return vectors

def get_data_vectors(data, features_to_keep):
    data_vectors = []
    data_answers = []
    for row in data:
        try:
            feature_vector = [float(row[feature]) for feature in features_to_keep]
            data_vectors.append(feature_vector)
            data_answers.append(int(row["Survived"]))
        except Exception as e:
            pass
            # print("Row produced exception: {}".format(row))

    return data_vectors, data_answers

def get_name_prefix(name):
    tokens = name.split()

    for token in tokens:
        if token[-1] == ".":
            return token

    return None

# histogram(data, "SibSp", float)
# bar_chart(data, "Age", float)

def run(train_data, features_to_keep):
    train_x, train_y = get_data_vectors(train_data, features_to_keep)
    one_hot_train_y = one_hot(train_y, 2)

    train_x = preprocessing.scale(train_x)

    # tensorflow_model = TensorFlowModel(learning_rate, batch_size, epochs)
    # rbf_svm_model = SVMModel(train_x, train_y, kernel="rbf")
    # xgboost = XGBoostModel(train_x, train_y)
    # ensemble = DecisionTreeEnsemble(train_x, train_y)

    svm_parameters = {
        "kernel": ["rbf", "linear"],
        "C": [0.01, 0.1, 1.0, 10.0, 100.0]
    }

    ensemble_parameters = {
        "criterion": ["gini", "entropy"],
        "n_estimators": [10, 20, 50, 75, 100]
    }

    adaboost_parameters = {
        "n_estimators": [10, 20, 50, 75, 100],
        "learning_rate": [0.001, 0.01, 0.1, 1.0,]
    }

    xgboost_parameters = {
        "max_depth": [4, 8, 12, 16, 20]
    }

    scores = ["accuracy", "precision", "recall", "f1"]
    models = {
        "xgboost": GridSearchCV(XGBClassifier(), xgboost_parameters, cv=5, scoring=scores, refit="accuracy"),
        "adaboost": GridSearchCV(AdaBoostClassifier(), adaboost_parameters, cv=5, scoring=scores, refit="accuracy"), # Neural Net on TensorFlow
        "random_forests": GridSearchCV(RandomForestClassifier(), ensemble_parameters, cv=5, scoring=scores, refit="accuracy"), # Random Forests
        "svm": GridSearchCV(SVC(), svm_parameters, cv=5, scoring=scores, refit="accuracy") # SVM
    }

    for model in models:
        models[model].fit(train_x, train_y)
        for score in scores:
            mean_accuracies = models[model].cv_results_['mean_test_accuracy']
            mean_precision = models[model].cv_results_['mean_test_precision']
            mean_recall = models[model].cv_results_['mean_test_recall']
            mean_f1 = models[model].cv_results_['mean_test_f1']
            for acc, prec, rec, f1, params in zip(mean_accuracies, mean_precision, mean_recall, mean_f1, models[model].cv_results_['params']):
                print("{} (Accuracy: {}, Precision: {}, Recall: {}, F1: {}) for {}".format(model, acc, prec, rec, f1, params))
            print("=====================================")

    # return results

    # models = {
    #     "TensorFlow Model": tensorflow_model,
    #     "RBF SVM Model": rbf_svm_model,
    #     "XGBoost": xgboost
    # }

    # return [(model, models[model].accuracy(test_x, test_y)) for model in models]

if __name__ == "__main__":
    # Variables: ['Fare', 'Ticket', 'Survived', 'Pclass', 'Sex', 'Embarked', 'Name', 'PassengerId', 'Cabin', 'SibSp', 'Age', 'Parch']

    # Read the data from CSV
    with open("data/train.csv") as train_csv:
        data = [row for row in csv.DictReader(train_csv)]

    # Create a new feature named Prefix - contains the prefix of each name
    name_prefixes = set()
    for row in data:
        if row["Name"] != "" or row["Name"] is not None:
            prefix = get_name_prefix(row["Name"])

            if prefix is not None:
                name_prefixes.add(prefix)

    # Map which casts sex to a categorical numerical variable
    sex_map = {"male": 0, "female": 1}

    # Map with all possible prefixes
    name_prefix_map = {}
    i = 0
    for prefix in name_prefixes:
        name_prefix_map[prefix] = i
        i += 1

    # For each row, cast sex to numerical variable and add the new Prefix feature
    # Ignore rows that raise an Exception (due to missin data)
    for row in data:
        try:
            row["Sex"] = sex_map[row["Sex"]]
            if row["Name"] is not None:
                row["Prefix"] = name_prefix_map[get_name_prefix(row["Name"])]
        except Exception as e:
            pass

    # Shuffle the data
    seed = 2373
    data = shuffle(data, random_state=seed)

    # Dict which will hold accuracies per feature list and model
    accuracies = {}

    # Feature lists that we will use for training
    features_lists = [
        ["Sex", "Age", "Pclass", "Fare", "Parch", "SibSp", "Prefix"],
    ]

    run(data, features_lists[0])