import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from decision_tree import *
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from prettytable import PrettyTable
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
    X_train = training_data.drop("Y", axis=1)
    X_test = testing_data.drop("Y", axis=1)

    # loo = LeaveOneOut()
    param_grid = {'max_depth': [10], 'min_samples_leaf': [2], 'min_samples_split': [2]}
    clf = GridSearchCV(
        DecisionTreeClassifier(criterion="entropy"), param_grid=param_grid,
        cv=LeaveOneOut(), verbose=True)
    # clf = DecisionTreeClassifierImpl(max_depth=10, min_samples_leaf=2, min_samples_split=2, criterion="entropy")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # cm = confusion_matrix(y_test, y_pred)
    # ac = accuracy_score(y_test, y_pred)
    # print(cm)
    # print(ac)

    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    MCCs = []
    auROCs = []
    auPRCs = []

    accuracy_scores.append(accuracy_score(y_true=y_test, y_pred=y_pred))
    f1_scores.append(f1_score(y_true=y_test, y_pred=y_pred))
    recall_scores.append(recall_score(y_true=y_test, y_pred=y_pred))
    precision_scores.append(precision_score(y_true=y_test, y_pred=y_pred))
    MCCs.append(matthews_corrcoef(y_true=y_test, y_pred=y_pred))
    auROCs.append(roc_auc_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 1]))
    auPRCs.append(average_precision_score(y_true=y_test, y_score=clf.predict_proba(X_test)[:, 0]))

    table = PrettyTable()
    column_names = ['Accuracy', 'auROC', 'auPRC', 'recall', 'precision', 'f1', 'MCC']
    table.add_column(column_names[0], np.round(accuracy_scores, 4))
    table.add_column(column_names[1], np.round(auROCs, 4))
    table.add_column(column_names[2], np.round(auPRCs, 4))
    table.add_column(column_names[3], np.round(recall_scores, 4))
    table.add_column(column_names[4], np.round(precision_scores, 4))
    table.add_column(column_names[5], np.round(f1_scores, 4))
    table.add_column(column_names[6], np.round(MCCs, 4))

    print(table)

    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, plot_confusion_matrix
    plot_confusion_matrix(clf, X_test, y_test)
    plt.title("Logistic Regression Model - Confusion Matrix")
    plt.xticks(range(2), ["Admit", "Discharge"], fontsize=8)
    plt.yticks(range(2), ["Admit", "Discharge"], fontsize=8)
    plt.show()

    from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
    plot_precision_recall_curve(clf, X_test, y_test, pos_label=1)

    plot_roc_curve(clf, X_test, y_test, pos_label=1)


if __name__ == '__main__':
    BRCA = load_data()
    train(BRCA)
