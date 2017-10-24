from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def variance_kbest(train_x, train_y, test_x, k=None):
    if k is None:
        k = len(train_x[0])

    kbest = SelectKBest(chi2, k=k)

    kbest.fit(train_x, train_y)

    return kbest.transform(train_x), kbest.transform(test_x)

