from models.tree.dt import DecisionTree
from models.tree.rf import RandomForest
from models.dtb.sgd import SGD

"""
Class representing all models. Contains objects of all models and provides methods to train and evaluate all models.
"""
class Models:
    """ 
    Constructor: initialise all models
    """
    def __init__(self, data):
        self.models = [
            DecisionTree(data), 
            RandomForest(data), 
            SGD(data)
        ];

    """
    Train all models
    """
    def train(self):
        [model.train() for model in self.models];

    """
    Get confusion matrices for all models
    """
    def get_confusion_matrices(self):
        return [model.confusion_matrix for model in self.models];

    """
    Get F1 scores
    """
    def get_f1_scores(self):
        return [model.f1 for model in self.models];

    """
    Get decision thresholds
    """
    def get_decision_thresholds(self):
        return [model.threshold for model in self.models];

    """
    Tune models
    """
    def tune(self):
        [model.tune_params(model.param_grid) for model in self.models];
