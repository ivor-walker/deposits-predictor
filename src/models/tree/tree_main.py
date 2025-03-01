from src.models.base import BaseClassifier

"""
Parent class for tree-based models
"""
class TreeClassifier(BaseClassifier):
    """
    Constructor: pass data to base class and set parameter grid
    """
    def __init__(self, data):
        super().__init__(data);
    
    """
    Set parameter grid for hyperparameter tuning
    """
    def _set_param_grid(self,
        param_grid = None
    ):
        if param_grid is None:
            param_grid = self.param_grid;
        
        # Set random state to ensure reproducability
        param_grid["random_state"] = [42];

        self.param_grid = param_grid;
