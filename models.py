from models import *

"""
Class representing all models. Contains objects of all models and provides methods to train and evaluate all models.
"""
class Models:
    """
    Constructor: initialise all models
    """
    def __init__(self, data):
        self.models = {
             
        };

    """
    Train all models
    """
    def train(self):
        for model in self.models.values(): 
            model.train(); 

    """
    Get confusion matrices for all models
    """
    def calculate_confusion_matrix(self):
        return {
            name: model.calculate_confusion_matrix() for name, model in self.models.items()
        };

    """
    Calculate F1 scores
    """
    def calculate_f1_score(self):
        return {
            name: model.calculate_f1_score() for name, model in self.models.items()
        };
