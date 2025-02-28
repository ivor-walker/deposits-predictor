from probability_classifier import ProbabilityClassifier

"""
Class for distance to boundary methods (e.g SGD)
"""
class DistanceToBoundary(ProbabilityClassifier):
    """
    Constructor: pass data to parent class
    """
    def __init__(self, data):
        super().__init__(data);

    """
    Implement predict abstract class for distance to boundary methods
    """
    def predict(self, train_X):
        return self.model.decision_function(train_X);
