import numpy as np

from abc import ABC, abstractmethod

"""
Base class for a model that can be trained and produces predictions. This class provides public wrappers for training and prediction, and handles threshold optimisation and model evaluation.
"""

class Model:
    """
    Constructor: set data
    """
    def __init_(self, data,
        requested_data_type = "insensitive",
        default_threshold = 0.5
    ):
        # Set classification threshold
        self.threshold = default_threshold;

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
    Public method to train the model and calculate optimal threshold
    """
    def train(self):
        self._train();
        self._calculate_optimal_threshold();

    """
    Abstract protected method to actually train the model using train_X  
    """
    @abstractmethod
    def _train(self):
        raise NotImplementedError("Training not implemented");

    """
    Public method to produce true/false based on probability threshold
    @param data: data to predict on
    @return: predicted labels
    """
    def predict(self, data):
        return self._predict(data) > self.threshold;
    
    """
    Abstract protected method to actually predict probability of "yes" using the model
    @return: predicted probabilities
    """
    @abstractmethod
    def _predict(self, data):
        raise NotImplementedError("Prediction not implemented");
    
    """
    Calculate optimal threshold values
    @param optimisation_method: method to use for optimisation. f1 is default as data contains class imbalance
    @param n: number of thresholds to consider between 0 and 1
    """
    def calculate_optimal_threshold(self,
        optimisation_method = "f1",
        n = 100
    ):
        # Calculate probabilities for test data
        self._predict_test();
        
        # Calculate possible thresholds
        possible_thresholds = np.linspace(0, 1, n);
        
        if optimisation_method == "f1":
            # Get threshold that maximises f1
            calculate_precision_recall(possible_thresholds);
            calculate_f1s();
            self.threshold = possible_thresholds[np.argmax(self.f1s)];
        
        elif optimisation_method == "youden":
            # Get threshold that maximises Youden's J
            _calculate_sensitivity_specificity(possible_thresholds);
            _calculate_youdens();
            self.threshold = possible_thresholds[np.argmax(self.youdens)];

    """
    Calculate probabilities for test data 
    @return: predicted probabilities
    """
    def _predict_test(self):
        self.pred_y = self._predict(self.test_X);
   
    """
    Calculate F1 scores for all precisions and recalls
    """
    def _calculate_f1s(self):
        # Calculate F1 scores of all precisions and recalls
        self.f1s = [self._calculate_f1(precision, recall) for precision, recall in zip(self.precisions, self.recalls)]; 

    """
    Calculate F1 score for a single given threshold
    @param precision: precision at given threshold
    @param recall: recall at given threshold
    """
    def _calculate_f1(self, precision, recall):
        return 2 * precision * recall / (precision + recall);

    """
    Calculate precision and recall for all given thresholds
    @param possible_thresholds: array of possible thresholds
    """
    def _calculate_precision_recall(self, possible_thresholds):
        self.precisions = [self._calculate_precision(threshold) for threshold in possible_thresholds];
        self.recalls = [self._calculate_recall(threshold) for threshold in possible_thresholds];
    
    """
    Calculate precision for a single given threshold
    @param threshold: threshold to calculate
    """
    def _calculate_precision(self, threshold):
        true_positives = np.sum((self.pred_y > threshold) & (self.true_y == 1));
        false_positives = np.sum((self.pred_y > threshold) & (self.true_y == 0));
        return true_positives / (true_positives + false_positives);

    """
    Calculate recall for a single given threshold
    @param threshold: threshold to calculate for
    """
    def _calculate_recall(self, threshold):
        true_positives = np.sum((self.pred_y > threshold) & (self.true_y == 1));
        false_negatives = np.sum((self.pred_y <= threshold) & (self.true_y == 1));
        return true_positives / (true_positives + false_negatives);

    """
    Calculate Youden's J statistic for all given sensitivities and specificities
    """
    def _calculate_youdens(self):
        self.youdens = [self._calculate_youden(sensitivity, specificity) for sensitivity, specificity in zip(self.sensitivities, self.specificities)];
    
    """
    Calculate Youden's J statistic for a single given sensitivity and specificity
    @param sensitivity: sensitivity to calculate
    @param specificity: specificity to calculate
    """
    def _calculate_youden(self, sensitivity, specificity):
        return sensitivity + specificity - 1;

    """
    Calculate sensitivity and specificity for all given thresholds
    @param possible_thresholds: array of possible thresholds
    """
    def _calculate_sensitivity_specificity(self, possible_thresholds):
        self.sensitivities = [self._calculate_recall(threshold) for threshold in possible_thresholds];
        self.specificities = [self._calculate_specificity(threshold) for threshold in possible_thresholds];

    """
    Calculate specificity for a single given threshold
    @param threshold: threshold to calculate for
    """
    def _calculate_specificity(self, threshold):
        true_negatives = np.sum((self.pred_y <= threshold) & (self.true_y == 0));
        false_positives = np.sum((self.pred_y > threshold) & (self.true_y == 0));
        return true_negatives / (true_negatives + false_positives);

    """
    Return confusion matrix for the optimal threshold
    @return: confusion matrix
    """
    def calculate_confusion_matrix(self):
        return [
            np.sum((self.pred_y > self.threshold) & (self.true_y == 1)),
            np.sum((self.pred_y <= self.threshold) & (self.true_y == 1)),
            np.sum((self.pred_y > self.threshold) & (self.true_y == 0)),
            np.sum((self.pred_y <= self.threshold) & (self.true_y == 0))
        ]; 

    """
    Calculate F1 score for the optimal threshold
    """
    def calculate_f1(self):
        self.precision = self._calculate_precision(self.threshold);
        self.recall = self._calculate_recall(self.threshold);
        self.f1 = self._calculate_f1(self.precision, self.recall);

    """
    Calculate Youden's J statistic for the optimal threshold
    """
    def calculate_youden(self):
        self.sensitivity = self._calculate_recall(self.threshold);
        self.specificity = self._calculate_specificity(self.threshold);
        self.youden = self._calculate_youden(self.sensitivity, self.specificity);

