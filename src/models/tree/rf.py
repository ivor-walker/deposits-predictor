from probability_classifier import ProbabilityClassifier

from sklearn.ensemble import RandomForestClassifier

"""
Class representing a random forest classifier
"""
class RandomForest(ProbabilityClassifier):
    """
    Constructor: initialise the model
    """
    def __init__(self, data):
        super().__init__(data);
        self.model = RandomForestClassifier();
