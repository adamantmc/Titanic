import csv
import numpy as np

def get_data_vectors(data, features_to_keep=None):
    # Convert data to x, y vectors, ignoring missing values

    data_vectors = []
    data_answers = []
    for row in data:
        try:
            feature_vector = [float(row[feature]) for feature in features_to_keep]

            data_vectors.append(feature_vector)
            data_answers.append(int(row["Survived"]))
        except Exception as e:
            pass

    return data_vectors, data_answers

def get_data(data_file):
    # Get data from CSV file, process them and return
    data = read_data(data_file)
    return process_data(data)

def read_data(data_file):
    # Read the data from CSV
    with open(data_file) as data_csv:
        data = [row for row in csv.DictReader(data_csv)]

    return data

def get_name_prefix(name):

    # Returns the prefix of a name

    tokens = name.split()

    for token in tokens:
        # Prefixes end in ., like Mrs. Smith
        if token[-1] == ".":
            return token

    return None

def process_data(data):
    # Variables:
    #   ['Fare', 'Ticket', 'Survived', 'Pclass',
    #   'Sex', 'Embarked', 'Name', 'PassengerId',
    #   'Cabin', 'SibSp', 'Age', 'Parch']

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
    # Ignore rows that raise an Exception (due to missing data)
    for row in data:
        try:
            row["Sex"] = sex_map[row["Sex"]]
            if row["Name"] is not None:
                row["Prefix"] = name_prefix_map[get_name_prefix(row["Name"])]
        except Exception as e:
            pass

    return data

