from src.models.tree.dt import DecisionTree
from src.models.tree.rf import RandomForest
from src.models.dtb.sgd import SGD

from tqdm import tqdm

import json

"""
Class representing all models. Contains objects of all models and provides methods to train and evaluate all models.
"""
class Models:
    """ 
    Constructor: initialise all models
    """
    def __init__(self, data):
        self.models = [
            SGD(data),
            DecisionTree(data), 
            RandomForest(data)
        ];
    
    """
    Tune all models' hyperparameters with a tqdm progress bar
    """
    def tune(self,
        save = True,
        load = False
    ):
        if load:
            self._load_params();
            return;

        for model in tqdm(self.models):
            model.tune();
        
        if save:
            self._save_params();
    
    """
    Save hyperparameters to JSON
    """
    def _save_params(self,
        path = "src/hyperparameters.json"
    ):
        best_params = [model.best_params for model in self.models];        
        with open(path, "w") as f:
            json.dump(best_params, f);

    """
    Load hyperparameters from JSON
    """
    def _load_params(self,
        path = "src/hyperparameters.json"
    ):
        best_params = None;
        with open(path, "r") as f:
            best_params = json.load(f);
        
        [model.set_params(params) for model, params in zip(self.models, best_params)];

    """
    Train all models with a tqdm progress bar
    """
    def train(self):
        for model in tqdm(self.models):
            model.train();

    """
    Evaluate all models
    """
    def evaluate(self):
        [model.evaluate() for model in self.models];
    
    """
    Get f1 scores of all models
    """
    def get_f1s(self):
        return [model.f1 for model in self.models];

    """
    Get confusion matrices of all models
    """
    def get_confusion_matrices(self):
        return [model.confusion_matrix for model in self.models];

    """
    Get ROC curve integral of all models
    """
    def get_roc_integrals(self):
        return [model.roc_integral for model in self.models];

    """
    Get test prediction ranges
    """
    def get_pred_ranges(self):
        return [[min(model.test_pred_y), max(model.test_pred_y)] for model in self.models]; 
