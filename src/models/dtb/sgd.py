from models.dtb.dtb_main import DTBClassifier

from sklearn.linear_model import SGDClassifier

"""
Class representing a Stochastic Gradient Descent (SGD) model.
"""
class SGD(DTBClassifier):
    """
    Constructor: call parent constructor and initialise model
    """
    def __init__(self, data):
        super().__init__(data);
        self.name = "SGD";
        self.model = SGDClassifier();

        self.param_grid = {
            'loss': ['hinge', 'log', 'modified_huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [0.0001, 0.001, 0.01],
            'max_iter': [1000, 2000],
            'tol': [1e-3, 1e-4]
        };
