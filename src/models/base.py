import numpy as np

"""
Base class for a model that can be trained and produces a probability prediction. This class handles fitting, threshold optimisation and evaluation. It also provides public wrappers for prediction.
"""

class BaseClassifier:
    """
    Constructor: set data and default threshold
    """
    def __init__(self, data,
        requested_data_type = "insensitive",
        default_threshold = 0.5
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

        self.default_threshold = default_threshold;
    
    """
    Train the model 
    """
    def train(self):
        self.model.fit(self.train_X, self.train_y);
        self.pred_y = self.predict(self.test_X);
        self._calculate_optimal_threshold();
    
    """
    Predict labels for test data
    """
    def predict(self, test_X):
        return self._predict(test_X) >= self.threshold;

    """
    Abstract method to predict probabilities
    Implemented by group of model classes (e.g tree)
    @param test_X: data to generate predictions for
    """
    @abstractmethod
    def _predict(self, test_X):
        raise NotImplementedError("Predict method not implemented");

    """
    Calculate optimal threshold values
    @param optimisation_method: method to use for optimisation. f1 is default as data contains class imbalance
    @param n: number of thresholds to consider between 0 and 1
    """
    def _calculate_optimal_threshold(self,
        optimisation_method = "f1",
        n = 100
    ):
        # Calculate possible thresholds
        possible_thresholds = np.linspace(0, 1, n);
        
        # Get confusion matrices for each threshold
        confusion_matrices = [self.calculate_confusion_matrix(threshold) for threshold in possible_thresholds];

        if optimisation_method == "f1":
            f1_optimisation(confusion_matrices); 

        elif optimisation_method == "youden":
            youden_optimisation(confusion_matrices); 

        else:
            raise ValueError("Invalid optimisation method");
    
    """
    Optimise threshold using F1 score
    @param confusion_matrices: confusion matrices for each threshold
    """
    def f1_optimisation(self, confusion_matrices):
        # Calculate precisions and recalls for each confusion matrix
        self.precisions = [self._calculate_precision(confusion_matrix) for confusion_matrix in self.confusion_matrices]; 
        self.recalls = [self._calculate_recall(confusion_matrix) for confusion_matrix in self.confusion_matrices];

        # Calculate f1 scores for each precision and recall
        self.f1s = np.array(
            [self._calculate_f1(precision, recall) for precision, recall in zip(self.precisions, self.recalls)]
        );
        
        # Find threshold with maximum f1
        max_f1_idx = np.argmax(self.f1s);
        self.threshold = possible_thresholds[max_f1_idx];

        # Get precision, recall and f1 for this threshold
        self.precision = self.precisions[max_f1_idx];
        self.recall = self.recalls[max_f1_idx];
        self.f1 = self.f1s[max_f1_idx]; 

    """ 
    Optimise threshold using Youden's J statistic
    @param confusion_matrices: confusion matrices for each threshold
    """
    def youden_optimisation(self, confusion_matrices):
        # Calculate sensitivities and specificities for each confusion matrix 
        self.sensitivities = [self._calculate_recall(confusion_matrix) for confusion_matrix in self.confusion_matrices];
        self.specificities = [self._calculate_specificity(confusion_matrix) for confusion_matrix in self.confusion_matrices];

        # Calculate Youden's J statistic for each sensitivity and specificity
        self.youdens = np.array(
            [self._calculate_youden(sensitivity, specificity) for sensitivity, specificity in zip(self.sensitivities, self.specificities)]
        );

        # Find threshold with maximum Youden's J statistic
        max_youden_idx = np.argmax(self.youdens);
        self.threshold = possible_thresholds[max_youden_idx];
        
        # Get sensitivity, specificity and Youden's J statistic for this threshold
        self.sensitivity = self.sensitivities[max_youden_idx];
        self.specificity = self.specificities[max_youden_idx];
        self.youden = self.youdens[max_youden_idx];
 
    """
    Create a confusion matrix using a threshold to determine true/false
    @return: confusion matrix
    """
    def calculate_confusion_matrix(self,
        threself.calhold = None 
    ):
        if threshold is None:
            threshold = self.threshold;

        return {
            "true_positive": np.sum(np.logical_and(self.true_y == 1, self.pred_y >= threshold)),
            "true_negative": np.sum(np.logical_and(self.true_y == 0, self.pred_y < threshold)),
            "false_positive": np.sum(np.logical_and(self.true_y == 0, self.pred_y >= threshold)),
            "false_negative": np.sum(np.logical_and(self.true_y == 1, self.pred_y < threshold))
        };
    
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
    def _calculate_recall(self, confusion_matrix):
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
    def _calculate_specificity(self, confusion_matrix):
        return confusion_matrix["true_negative"] / (confusion_matrix["true_negative"] + confusion_matrix["false_positive"]);
