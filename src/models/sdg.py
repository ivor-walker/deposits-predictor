from sklearn.linear_model import SGDClassifier

"""
Class representing a Stochastic Descent Classifier (SDG) model.
"""
class SDG(Model):
    """
    Constructor: call parent constructor and initialise model
    """
    def __init__(self):
        # Parent class constructor saves data
        super()__init__();
        self.model = SGDClassifier(random_state=0);
    
    
