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

        self.model_names = [model.name for model in self.models];
    
    """
    Tune all models' hyperparameters with a tqdm progress bar
    """
    def tune(self,
        load_path = None,
        save_path = None,
    ):
        if load_path:
            self._load_params(load_path);
            return;

        for model in tqdm(self.models):
            model.tune();
        
        if save_path:
            self._save_params(save_path);
    
    """
    Save hyperparameters to JSON
    @param path: file to save hyperparameters to
    """
    def _save_params(self, path):
        best_params = self.get_best_params(wrap = False);
        
        with open(path, "w") as f:
            json.dump(best_params, f);

    """
    Load hyperparameters from JSON
    @param path: file to load hyperparameters from
    """
    def _load_params(self, path):
        params = None;
        with open(path, "r") as f:
            params = json.load(f);
        
        [model.set_params(params) for model, params in zip(self.models, params)];
    
    """
    Get hyperparameter grid of all models
    """
    def get_param_grids(self):
        param_grids = [model.param_grid for model in self.models];
        return self.wrap_names(param_grids);

    """
    Helper function to wrap model names around model attributes
    """
    def wrap_names(self, attributes):
        return [
            {name: attribute}
            for name, attribute in zip(self.model_names, attributes)
        ];

    """
    Get set hyperparameters of all models
    """
    def get_params(self):
        params = [model.params for model in self.models];
        return self.wrap_names(params);
    
    """
    Get best hyperparameters chosen by CV of all models
    @param wrap: wrap model names around hyperparameters
    """
    def get_best_params(self, wrap = True):
        params = [model.best_params for model in self.models];
        
        if wrap:
            return self.wrap_names(params);

        return params;

    """
    Train all models with a tqdm progress bar
    """
    def train(self):
        for model in tqdm(self.models):
            model.train();
    
    """
    Get selected decision thresholds of all models
    """
    def get_thresholds(self):
        thresholds = [model.threshold for model in self.models];
        return self.wrap_names(thresholds);

    """
    Get training dataset F1 scores of all models
    """
    def get_train_f1s(self):
        train_f1s = [model.train_f1 for model in self.models];
        return self.wrap_names(train_f1s);

    """
    Get training dataset confusion matrices of all models
    """
    def get_train_confusion_matrices(self):
        train_confusion_matrices = [model.train_confusion_matrix for model in self.models];
        return self.wrap_names(train_confusion_matrices);

    """
    Get training dataset ROC curve integrals of all models
    """
    def get_train_roc_integrals(self):
        train_roc_integrals = [model.train_roc_integral for model in self.models];
        return self.wrap_names(train_roc_integrals);

    """
    Get training dataset prediction ranges of all models
    """
    def get_train_pred_ranges(self):
        train_pred_ranges = [[min(model.train_pred_y), max(model.train_pred_y)] for model in self.models]; 
        return self.wrap_names(train_pred_ranges);

    """
    Evaluate all models
    """
    def evaluate(self):
        [model.evaluate() for model in self.models];
    
    """
    Get f1 scores of all models
    """
    def get_f1s(self):
        f1s = [model.f1 for model in self.models];
        return self.wrap_names(f1s);

    """
    Get confusion matrices of all models
    """
    def get_confusion_matrices(self):
        confusion_matrices = [model.confusion_matrix for model in self.models];
        return self.wrap_names(confusion_matrices);

    """
    Get ROC curve integral of all models
    """
    def get_roc_integrals(self):
        roc_integrals = [model.roc_integral for model in self.models];
        return self.wrap_names(roc_integrals);

    """
    Get test prediction ranges
    """
    def get_pred_ranges(self):
        pred_ranges = [[min(model.test_pred_y), max(model.test_pred_y)] for model in self.models]; 
        return self.wrap_names(pred_ranges);

    """
    Get precisions and recalls at optimal thresholds of all models
    """
    def get_precisions_recalls(self):
        precisions_recalls = [[model.precision, model.recall] for model in self.models];
        return self.wrap_names(precisions_recalls);
