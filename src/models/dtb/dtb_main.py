from models.base import BaseClassifier

"""
Class for distance to boundary methods (e.g SGD)
"""
class DTBClassifier(BaseClassifier):
    """
    Constructor: pass data to parent class
    """
    def __init__(self, data):
        super().__init__(data);

    """
    Implement predict abstract class to produce a probability for distance to boundary methods
    """
    def _predict(self, train_X):
        return self.model.decision_function(train_X);
