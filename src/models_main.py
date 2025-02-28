from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.sdg import SDG

"""
Class representing all models. Contains objects of all models and provides methods to train and evaluate all models.
"""
class Models:
    """ 
    Constructor: initialise all models
    """
    def __init__(self, data):
        self.models = {
            "decision_tree": DecisionTree(data),
            "random_forest": RandomForest(data),
            "sdg": SDG(data)
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
    def get_confusion_matrices(self):
        return {
            name: model.confusion_matrix
            for name, model in self.models.items()
        };

    """
    Calculate F1 scores
    """
    def get_f1_scores(self):
        return {
            name: model.f1 
            for name, model in self.models.items()
        };
