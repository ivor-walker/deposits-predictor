from src.models.dtb.dtb_main import DTBClassifier

from sklearn.linear_model import SGDClassifier

"""
Class representing a Stochastic Gradient Descent (SGD) model.
"""
class SGD(DTBClassifier):
    """
    Constructor: initialise model, pass data and model to parent class and set parameter grid
    """
    def __init__(self, data):
        self.name = "SGD";
        model = SGDClassifier();
        super().__init__(data, model);

        self.param_grid = {
            'loss': ['hinge', 'log_loss', 'modified_huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [1000, 2000],
            'tol': [1e-3, 1e-4]
        };

        self._set_param_grid();
