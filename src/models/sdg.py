from base_classifier import BaseClassifier

from sklearn.linear_model import SGDClassifier

"""
Class representing a Stochastic Descent Classifier (SDG) model.
"""
class SDG(BaseClassifier):
    """
    Constructor: call parent constructor and initialise model
    """
    def __init__(self, data):
        super()__init__(data);
        self.model = SGDClassifier(random_state=0);
