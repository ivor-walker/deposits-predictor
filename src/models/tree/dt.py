from src.models.tree.tree_main import TreeClassifier

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
        
        # Parameters to create a more complex decision tree than default
        self.param_grid = {
            'max_depth': [None, 20, 40, 60],
            'max_leaf_nodes': [None, 20, 50, 100],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        };
