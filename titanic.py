import os
import pandas
import operator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GridSearchCV, train_test_split
from keras_model import KerasModel
from sklearn.utils import shuffle
from sklearn import preprocessing, metrics

from data_processing import *

def is_number(s):
    # I love this function
    try:
        float(s)
        return True
    except Exception:
        return False

def run_classifying(train_x, train_y, plots_folder="plots", tables_folder="tables"):
    # Create plots and tables folder

    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    if not os.path.exists(tables_folder):
        os.makedirs(tables_folder)

    # Define parameters for each model

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

    #From previous run, incrementing learning_rate only worsens it
    nn_parameters = {
        "layers": [[256], [256, 128], [256, 128, 64], [64], [64, 32], [64, 32, 16]],
        "activation_function": ["relu", "sigmoid"]
    }

    # Scores list
    scores = ["accuracy", "precision", "recall", "f1"]

    # Models dict - using GridSearchCV for grid search on parameters with Cross-Validation
    models = {
        "nn": GridSearchCV(KerasModel(), nn_parameters, cv=5, scoring=scores, refit="accuracy"), # Neural Net on Keras
        "adaboost": GridSearchCV(AdaBoostClassifier(), adaboost_parameters, cv=5, scoring=scores, refit="accuracy"), # Adaboost
        "random_forests": GridSearchCV(RandomForestClassifier(), ensemble_parameters, cv=5, scoring=scores, refit="accuracy"), # Random Forests
        "svm": GridSearchCV(SVC(), svm_parameters, cv=5, scoring=scores, refit="accuracy") # SVM
    }

    # Dict mapping model names to parameters
    model_params = {
        "nn": nn_parameters,
        "adaboost": adaboost_parameters,
        "random_forests": ensemble_parameters,
        "svm": svm_parameters
    }

    # We will build a dataframe for each model with its results
    model_dataframes = {}

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    for model in models:
        values = []

        # Fit each model, get its results and append to values list, along with the selected parameters
        models[model].fit(train_x, train_y)
        mean_accuracies = models[model].cv_results_['mean_test_accuracy']
        mean_precision = models[model].cv_results_['mean_test_precision']
        mean_recall = models[model].cv_results_['mean_test_recall']
        mean_f1 = models[model].cv_results_['mean_test_f1']

        for acc, prec, rec, f1, params in zip(mean_accuracies, mean_precision, mean_recall, mean_f1, models[model].cv_results_['params']):
            values.append([acc, prec, rec, f1, *[x[1] for x in sorted(params.items(), key=operator.itemgetter(0))]])
            print("{} (Accuracy: {}, Precision: {}, Recall: {}, F1: {}) for {}".format(model, acc, prec, rec, f1, params))

        # Create dataframe from values list
        df = pandas.DataFrame(values)
        df.columns = [*metrics, *sorted(model_params[model].keys())]
        model_dataframes[model] = df

        # Some magic to plot the results of a model in a single figure
        # Y axis is metric (or score)
        # X axis is the parameter with the most values
        # Each plot (or line) concerns a single value of the smallest parameter (the parameter with the least values)
        smaller_param = str(min(model_params[model], key=lambda x: len(model_params[model][x])))
        key_list = list(model_params[model].keys())
        key_list.remove(smaller_param)
        other_param = str(key_list[0])

        # Write dataframe to LaTeX table
        with open(os.path.join(tables_folder, "{}.txt".format(model)), "w") as f:
            f.write(df.to_latex())

        # For each metric, create a figure
        for metric in metrics:
            fig = plt.figure(figsize=(9.60, 5.40), dpi=100)
            ax = plt.subplot(111)

            # For each value of the smallest parameter, plot a line of its scores
            # in respect to the other parameter's values
            for val in model_params[model][smaller_param]:
                x = df.loc[df[smaller_param] == val][other_param]
                y = df.loc[df[smaller_param] == val][metric]

                # Some more pyplot magic in case our X axis values are not numbers (e.g. neural network layers)
                if not is_number(x.iloc[0]):
                    x_ticks = [str(elem) for elem in x]
                    x = range(len(x))
                    ax.set_xticks(x)
                    ax.set_xticklabels(x_ticks)

                plt.plot(
                    x, y,
                    label="{} = {}".format(str(smaller_param), str(val))
                )

            plt.xlabel(other_param)
            plt.ylabel(metric)
            plt.title(model)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=2)
            plt.savefig(os.path.join(plots_folder, "{}_{}.png".format(model, metric.lower())))

def run_clustering(x, y, tables_folder="tables"):
    if not os.path.exists(tables_folder):
        os.makedirs(tables_folder)

    clustering_params = {
        "linkage": ["ward", "complete", "average"],
        "affinity": ["euclidean", "manhattan", "cosine"]
    }

    results = []

    n_clusters = 2

    # Workaround - ward linkage can only be used with euclidean affinity
    ward_counter = 0
    for linkage in clustering_params["linkage"]:
        for affinity in clustering_params["affinity"]:
            if linkage == "ward":
                if ward_counter == 0:
                    affinity = "euclidean"
                    ward_counter += 1
                else:
                    break

            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity)
            labels = model.fit_predict(x)

            homogenity = metrics.homogeneity_score(labels_true=y, labels_pred=labels)
            completeness = metrics.completeness_score(labels_true=y, labels_pred=labels)
            silhouette = metrics.silhouette_score(x, labels)

            cluster_1_size = len([a for a in labels if a  == 0])
            cluster_2_size = len([a for a in labels if a  == 1])

            # Append to results list - will be used later on to create a dataframe
            results.append(
                [cluster_1_size, cluster_2_size, homogenity, completeness, silhouette, linkage, affinity]
            )

    # Build dataframe on results to expor to LaTeX table
    df = pandas.DataFrame(results)
    df.columns = ["Cluster 1 Size", "Cluster 2 Size", "Homogeneity", "Completeness", "Silhouette Score", "Linkage", "Affinity"]

    with open(os.path.join(tables_folder, "clustering.txt"), "w") as f:
        f.write(df.to_latex())

if __name__ == "__main__":
    data = get_data("data/train.csv")

    # Shuffle the data
    seed = 42
    np.random.seed(seed)
    data = shuffle(data, random_state=seed)

    # Features to use
    features = ["Sex", "Age", "Pclass", "Fare", "Parch", "SibSp", "Prefix"]

    x, y = get_data_vectors(data, features)
    x = preprocessing.scale(x)

    run_classifying(x, y)
    run_clustering(x, y)

