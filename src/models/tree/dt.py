from models.tree import TreeClassifier

from sklearn.tree import DecisionTreeClassifier

"""
Class for decision tree classifier
"""
class DecisionTree(TreeClassifier):
    """
    Constructor: Initialise data and model
    @param data: Data to be used for training and testing
    """
    def __init__(self, data):
        super().__init__(data);
        self.model = DecisionTreeClassifier();
