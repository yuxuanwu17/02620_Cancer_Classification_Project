class Node:
    def __init__(self):
        # links to the left and right child nodes
        self.right = None
        self.left = None

        # derived from splitting criteria
        self.column = None
        self.threshold = None

        # probability for object inside the Node to belong for each of the given classes
        self.probas = None
        # depth of the given node
        self.depth = None

        # if it is the root Node or not
        self.is_terminal = False


class DecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples_leaf=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

        # Decision tree itself
        self.Tree = None

    def nodeProbas(self, y):
        '''
        Calculates probability of class in a given node
        '''

        pass

    def gini(self, probas):
        '''
        Calculates gini criterion
        '''

        pass

    def calcImpurity(self, y):
        '''
        Wrapper for the impurity calculation. Calculates probas first and then passses them
        to the Gini criterion
        '''
        pass

    def calcBestSplit(self, X, y):
        '''
        Calculates the best possible split for the concrete node of the tree
        '''

        pass

    def buildDT(self, X, y, node):
        '''
        Recursively builds decision tree from the top to bottom
        '''

        pass

    def fit(self, X, y):
        '''
        Standard fit function to run all the model training
        '''

        pass

    def predictSample(self, x, node):
        '''
        Passes one object through decision tree and return the probability of it to belong to each class
        '''

        pass

    def predict(self, X):
        '''
        Returns the labels for each X
        '''

        pass