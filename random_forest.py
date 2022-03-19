import numpy as np
import pandas as pd
from decision_tree import *

class RandomForest():
    def __init__(self, n_features, sample_size, n_trees=100, max_depth=10, min_leaf=5):
        """

        :param n_trees: number of uncorrelated trees we ensemble to create the random forest
        :param n_features:  the number of features to sample and pass onto the tree
        :param sample_size: the number of rows randomly selected and pass onto each tree
        :param depth:
        :param min_leaf:
        """

        # specify the n_features (sqrt, log2)
        self.n_features = n_features

        # init the para
        self.sample_size = sample_size
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.n_trees = n_trees

    def create_tree(self):
        """
        create a new decision tree by calling the constructor of class.
        Each tree receives a random subset of features (feature bagging) and a random set of rows (bagging trees)
        :return:
        """

        idx = np.random.permutation(len(self.y))[:self.sample_size]
        feat_idx = np.random.permutation(self.X.shape[1])[:self.n_features]

        DTrees = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_leaf)
        boots_X = self.X[idx, :]
        boots_y = self.y[idx]

        DTrees.fit(boots_X, boots_y)
        return DTrees

    def predict(self, X):
        pred = [t.predict(X) for t in self.trees]
        res = np.mean(pred,axis=0)
        return res

    def fit(self, X, y):

        if type(X) == pd.DataFrame:
            X = np.asarray(X)
            y = np.asarray(y)

        self.X = X
        self.y = y

        # create the tree
        # self.trees = [self.create_tree() for i in range(self.n_trees)]
        trees = []
        for i in range(self.n_trees):
            trees.append(self.create_tree())
            # print(i)
        self.trees = trees


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X, y, column_names = data['data'], data['target'], data['feature_names']
    X = pd.DataFrame(X, columns=column_names)
    X['target'] = y

    X, y = X.drop(columns='target'), X['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=44)

    model = RandomForest(n_features=4, sample_size=1000, max_depth=2, min_leaf=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    print(f'Accuracy for self built model {accuracy_score(y_val, y_pred)}')

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f'Accuracy for sklearn Random Forest {accuracy_score(y_val, y_pred)}')
