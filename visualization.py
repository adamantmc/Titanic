import matplotlib.pyplot as plt
import numpy as np

def get_rows(data, variable, type_converter):
    survived = []
    not_survived = []

    for row in data:
        variable_val = row[variable]
        survived_val = row["Survived"]
        if variable_val == "" or variable_val is None or survived == "" or survived is None:
            pass
        else:
            variable_val = type_converter(variable_val)
            survived_val = int(survived_val)

            if survived_val == 1:
                survived.append(variable_val)
            else:
                not_survived.append(variable_val)

    return survived, not_survived

def get_counts(data, variable, type_converter):
    survived_dict = {}
    not_survived_dict = {}

    for row in data:
        var_val = row[variable]
        survived_val = row["Survived"]

        if var_val == "" or var_val is None or survived_val == "" or survived_val is None:
            pass
        else:
            var_val = type_converter(var_val)
            survived_val = int(survived_val)

            if var_val not in survived_dict:
                survived_dict[var_val] = 0

            if var_val not in not_survived_dict:
                not_survived_dict[var_val] = 0

            if survived_val == 1:
                survived_dict[var_val] += 1
            else:
                not_survived_dict[var_val] += 1

    return survived_dict, not_survived_dict

def dict_to_lists(data):
    x = []
    y = []

    for item in data.items():
        x.append(item[0])
        y.append(item[1])

    return x, y

def histogram(data, variable, type_converter):

    survived, not_survived = get_rows(data, variable, type_converter)

    plt.subplot(2, 1, 1)
    plt.hist(survived, color="#03A9F4", alpha=0.5, label="Survived")
    plt.xlabel("{} of people who survived".format(variable))
    plt.ylabel('Count')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.hist(not_survived, color="#F44336", alpha=0.5, label="Did Not Survive")
    plt.xlabel("{} of people who did not survive".format(variable))
    plt.ylabel('Count')
    plt.grid(True)

    plt.legend()

    plt.show()

def plot(data, variable, type_converter):

    survived_dict, not_survived_dict = get_counts(data, variable, type_converter)

    survived_x, survived_y = dict_to_lists(survived_dict)
    not_survived_x, not_survived_y = dict_to_lists(not_survived_dict)

    plt.plot(survived_x, survived_y, color="#03A9F4", alpha=0.5, linestyle="solid")
    plt.plot(not_survived_x, not_survived_y, color="#F44336", alpha=0.5, linestyle="solid")
    plt.xlabel(variable)
    plt.ylabel('Count')
    plt.grid(True)

    plt.legend()

    plt.show()

def bar_chart(data, variable, type_converter):
    survived_dict, not_survived_dict = get_counts(data, variable, type_converter)

    survived_x, survived_y = dict_to_lists(survived_dict)
    not_survived_x, not_survived_y = dict_to_lists(not_survived_dict)

    plt.bar(survived_x, survived_y, label="Survived")
    plt.bar(not_survived_x, not_survived_y, label="Did not survive", bottom=survived_y)

    plt.xlabel(variable)
    plt.ylabel("Count")

    plt.legend()
    plt.show()