from src.models.tree.tree_main import TreeClassifier

from sklearn.ensemble import RandomForestClassifier

"""
Class representing a random forest classifier
"""
class RandomForest(TreeClassifier):
    """
    Constructor: initialise the model
    """
    def __init__(self, data):
        super().__init__(data);
        self.name = "Random Forest";
        self.model = RandomForestClassifier();

        self.param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        };
