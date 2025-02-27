from base_classifier import BaseClassifier

from sklearn.ensemble import RandomForestClassifier

"""
Class representing a random forest classifier
"""
class RandomForest(BaseClassifier):
    """
    Constructor: initialise the model
    """
    def __init__(self, data):
        super().__init__(data);
        self.model = RandomForestClassifier();
