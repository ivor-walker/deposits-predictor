from models.base import BaseClassifier

"""
Class for tree-based classifiers
"""
class TreeClassifier(BaseClassifier):
    """
    Constructor: pass data to base class
    """
    def __init__(self, data):
        super().__init__(data);

    """
    Implement abstract method to produce probability prediction using tree-based models 
    @param test_X: data to predict
    @return: probability prediction
    """
    def _predict(self, test_X):
        # Remove first column from result of predict_proba
        return self.model.predict_proba(test_X)[::,1];
