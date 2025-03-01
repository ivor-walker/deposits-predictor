from src.models.base import BaseClassifier

from sklearn.calibration import CalibratedClassifierCV

"""
Class for distance to boundary methods (e.g SGD)
"""
class DTBClassifier(BaseClassifier):
    """
    Constructor: pass data to base class and wrap model in Platts scaling model
    """
    def __init__(self, data, model):
        super().__init__(data);
        
        self._platt_wrap(model);
                
    """
    Wrap model in a Platts scaling model which turns decision_function distance into probabilities between 0 and 1 via a sigmoid function
    @param model: model to wrap
    """
    def _platt_wrap(self, model):
        if isinstance(model, CalibratedClassifierCV):
            return;

        # Hyperparameters for optimising Platts scaling
        self.platts_params = {
            'method': ['sigmoid'],
            'cv': [5]
        };

        self.model = CalibratedClassifierCV(model);

    """
    Modify hyperparameters to enable accessing of model wrapped in Platts scaling model
    @param param_grid: dictionary of hyperparameters targeting underlying model
    @param estimator_prefix: prefix for accessing underlying hyperparameters
    """
    def _set_param_grid(self,
        estimator_prefix = "estimator__"
    ):
        # Add prefix to all keys in param_grid
        self.param_grid = {
            (estimator_prefix + key) : value 
            for key, value in self.param_grid.items()
        };

        # Add Platts scaling hyperparameters
        for key, value in self.platts_params.items():
            self.param_grid[key] = value;

        # Add random state
        self.param_grid["estimator__random_state"] = [42];
