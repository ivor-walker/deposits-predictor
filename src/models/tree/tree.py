from prob import ProbabilityClassifier

"""
Class for tree-based classifiers
"""
class Tree(ProbabilityClassifier):
    """
    Constructor: pass data to parent class
    """
    def __init__(self, data):
        super().__init__(data);

    """
    Implement abstract method to predict using tree-based models 
    @param test_X
    """
    def predict(self, test_X):
        return self.model.predict_proba(test_X);
