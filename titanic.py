import csv
import operator
from models.tensorflow_model import *
from models.keras_model import *
from models.svm import *
from models.knn import *
from models.decision_tree import *
from models.decision_tree_ensemble import *
from analysis import *
from visualization import *
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

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

def run(train_data, test_data, features_to_keep):
    train_x, train_y = get_data_vectors(train_data, features_to_keep)
    test_x, test_y = get_data_vectors(test_data, features_to_keep)
    one_hot_train_y = one_hot(train_y, 2)

    # train_x, test_x = variance_kbest(train_x, train_y, test_x, k=5)

    learning_rate = 0.001
    batch_size = 16
    epochs = 20

    tensorflow_model = TensorFlowModel(train_x, one_hot_train_y, learning_rate, batch_size, epochs)
    keras_model = KerasModel(train_x, one_hot_train_y, learning_rate, batch_size, epochs)
    rbf_svm_model = SVMModel(train_x, train_y, kernel="rbf")
    knn_3_model = KNeighboursModel(train_x, train_y, 3)
    tree_entropy = DecisionTreeModel(train_x, train_y, "entropy")
    tree_gini = DecisionTreeModel(train_x, train_y, "gini")

    models = {
        "TensorFlow Model": tensorflow_model,
        "Keras Model": keras_model,
        "RBF SVM Model": rbf_svm_model,
        "3 Nearest Neighbours Model": knn_3_model,
        "Gini Decision Tree Model": tree_gini,
    }

    tree_entropy.visualize_png("entropy")
    tree_gini.visualize_png("gini")

    return [(model, models[model].accuracy(test_x, test_y)) for model in models]

if __name__ == "__main__":
    # Variables: ['Fare', 'Ticket', 'Survived', 'Pclass', 'Sex', 'Embarked', 'Name', 'PassengerId', 'Cabin', 'SibSp', 'Age', 'Parch']

    with open("data/train.csv") as train_csv:
        data = [row for row in csv.DictReader(train_csv)]

    name_prefixes = set()
    for row in data:
        if row["Name"] != "" or row["Name"] is not None:
            prefix = get_name_prefix(row["Name"])

            if prefix is not None:
                name_prefixes.add(prefix)

    sex_map = {"male": 0, "female": 1}
    name_prefix_map = {}
    i = 0
    for prefix in name_prefixes:
        name_prefix_map[prefix] = i
        i += 1

    for row in data:
        try:
            row["Sex"] = sex_map[row["Sex"]]
            if row["Name"] is not None:
                row["Prefix"] = name_prefix_map[get_name_prefix(row["Name"])]
        except Exception as e:
            pass

    seed = 2373
    data = shuffle(data, random_state=seed)
    n_folds = 5

    kf = KFold(n_splits=n_folds)

    accuracies = {}

    features_lists = [
        ["Sex", "Age"],
        ["Sex", "Age", "Pclass", "Fare"],
        ["Sex", "Age", "Pclass", "Fare", "Parch", "Prefix"],
    ]

    for features_to_keep in features_lists:
        features_to_keep_str = ", ".join(features_to_keep)

        model_acc_dict = {}
        for train_data, test_data in kf.split(data):
            train_data = [data[index] for index in train_data]
            test_data = [data[index] for index in test_data]
            results = run(train_data, test_data, features_to_keep)

            for tup in results:
                if tup[0] not in model_acc_dict:
                    model_acc_dict[tup[0]] = 0
                model_acc_dict[tup[0]] += tup[1]

        for model in model_acc_dict:
            model_acc_dict[model] /= n_folds

        accuracies[features_to_keep_str] = model_acc_dict

    for features in accuracies:
        print("Accuracies for {}:".format(features))
        print(sorted(accuracies[features].items()))