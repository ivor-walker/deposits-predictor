from models.base import BaseClassifier

from sklearn.calibration import CalibratedClassifierCV

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
    Wrap given model in a Platts scaling model to get probabilities between 0 and 1
    """
    def platt_wrap(self, model):
        if not isinstance(model, CalibratedClassifierCV):
            self.model = CalibratedClassifierCV(model, cv=5, method='sigmoid')
