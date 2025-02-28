from models.tree.tree_main import TreeClassifier

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
        self.name = "Decision Tree";
        self.model = DecisionTreeClassifier();

        self.param_grid = {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        };
