from sklearn.tree import DecisionTreeClassifier

"""
Class for decision tree classifier
"""
class DecisionTree(Model):
    def __init__(self):

        self.model = DecisionTreeClassifier()
