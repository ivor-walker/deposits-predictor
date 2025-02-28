import numpy as np

from abc import ABC, abstractmethod

"""
Base class for a model that can be trained and produces predictions. This class provides public wrappers for training and prediction, and handles model evaluation.
"""

class BaseClassifier:
    """
    Constructor: set data
    """
    def __init_(self, data,
        requested_data_type = "insensitive",
    ):
        # Set X based on requested type
        if requested_data_type == "insensitive": 
            self.train_X = data.insensitive_train_X;
            self.test_X = data.insensitive_test_X;
        elif requested_data_type == "sensitive":
            self.train_X = data.sensitive_train_X;
            self.test_X = data.sensitive_test_X;
        else:
            raise ValueError("Unknown data type requested");
        
        self.train_y = data.train_y;
        self.true_y = data.test_y;

    """
    Train the model 
    """
    def train(self):
        self.model.fit(self.train_X, self.train_y);
        self.pred_y = self.model.predict(self.test_X);

    """
    Predict on a given dataset 
    @param data: dataframe to predict on
    @return: predicted labels
    """
    def predict(self, test_X):
        self.model.predict(test_X);

    """
    Calculate a confusion matrix 
    @return: confusion matrix
    """
    def calculate_confusion_matrix(self):
        return {
            "false_positive": np.sum(self.pred_y == 1) & (self.true_y == 0));
            "false_negative": np.sum(self.pred_y == 0) & (self.true_y == 1));
            "true_positive": np.sum(self.pred_y == 1) & (self.true_y == 1));
            "true_negative": np.sum(self.pred_y == 0) & (self.true_y == 0));
        };
    
    """
    Calculate F1 score of the model
    """
    def calculate_f1(self):
        confusion_matrix = self.calculate_confusion_matrix();

        precision = self._calculate_precision(confusion_matrix);
        recall = self._calculate_recall(confusion_matrix);

        return self._calculate_f1(precision, recall);

    """
    Calculate F1 score for a given precision and recall 
    @param precision: precision to calculate for
    @param recall: recall to calculate for
    """
    def _calculate_f1(self, precision, recall):
        return 2 * precision * recall / (precision + recall);

    """
    Calculate precision for a given confusion matrix
    @param confusion_matrix: confusion matrix to calculate for
    """
    def _calculate_precision(self, confusion_matrix):
        return confusion_matrix["true_positive"] / (confusion_matrix["true_positive"] + confusion_matrix["false_positive"]);

    """
    Calculate recall for a given confusion matrix
    @param confusion_matrix: confusion matrix to calculate for
    """
    def _calculate_recall(self, threshold):
        return confusion_matrix["true_positive"] / (confusion_matrix["true_positive"] + confusion_matrix["false_negative"]);
        
    """
    Calculate Youden's J statistic for a single given sensitivity and specificity
    @param sensitivity: sensitivity to calculate
    @param specificity: specificity to calculate
    """
    def _calculate_youden(self, sensitivity, specificity):
        return sensitivity + specificity - 1;
        
    """
    Calculate specificity for a given confusion matrix
    @param confusion_matrix: confusion matrix to calculate for
    """
    def _calculate_specificity(self, threshold):
        return confusion_matrix["true_negative"] / (confusion_matrix["true_negative"] + confusion_matrix["false_positive"]);
