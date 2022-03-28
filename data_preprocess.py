import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from decision_tree import *
import warnings


def load_data():
    BRCA = pd.read_csv("data/BRCA.txt", delimiter="\t")
    columns = BRCA.columns
    # y labels
    Y = [1 if i[-3:] == '11A' else 0 for i in columns]
    BRCA = BRCA.T
    BRCA['Y'] = Y

    return BRCA


def train(df):
    training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25, shuffle=True)
    y_train = training_data['Y']
    y_test = testing_data['Y']
    X_train = StandardScaler().fit_transform(training_data.drop("Y", axis=1))
    X_test = StandardScaler().fit_transform(testing_data.drop("Y", axis=1))

    clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=2, criterion="entropy")
    # clf = DecisionTreeClassifierImpl(max_depth=10, min_samples_leaf=2, min_samples_split=2, criterion="entropy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test, y_pred)
    print(cm)
    print(ac)


if __name__ == '__main__':
    BRCA = load_data()
    train(BRCA)
