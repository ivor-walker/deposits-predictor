import numpy as np

from sklearn.model_selection import GridSearchCV

"""
Base class for a model that can be trained and produces a probability prediction. This class handles fitting, threshold optimisation and evaluation. It also provides public wrappers for prediction.
"""

class BaseClassifier:
    """
    Constructor: set data and default threshold
    """
    def __init__(self, data,
        requested_data_type = "insensitive",
        default_threshold = 0.5,
    ):
        # Set X based on requested type of data
        if requested_data_type == "insensitive": 
            self.train_X = data.insensitive_train_X;
            self.validate_X = data.insensitive_validate_X;
            self.test_X = data.insensitive_test_X;
        elif requested_data_type == "sensitive":
            self.train_X = data.sensitive_train_X;
            self.validate_X = data.insensitive_validate_X;
            self.test_X = data.sensitive_test_X;
        else:
            raise ValueError("Unknown data type requested");
        
        # Set ys
        self.train_y = data.train_y;
        self.validate_y = data.validate_y;
        self.true_y = data.test_y;
        
        # Set default decision threshold
        self.threshold = default_threshold;
    
    """
    Train the model 
    """
    def train(self):
        # Fit model
        self.model.fit(self.train_X, self.train_y);
        
        # Store probability predictions for test data
        self.pred_y = self._predict(self.test_X);
        
        # Get optimal threshold using probabilities
        self._calculate_optimal_threshold();
    
    """
    Produce binary predictions for given data 
    @param test_X: data to generate predictions for
    @return: binary predictions
    """
    def predict(self, test_X):
        return self._predict(test_X) >= self.threshold;

    """
    Produce probability predictions for given data
    @param test_X: data to generate predictions for
    @return: probability predictions
    """
    def _predict(self, test_X):
        # Remove first column from result of predict_proba
        return self.model.predict_proba(test_X)[::,1];

    """
    Tune hyperparameters 
    @param param_grid: possible hyperparameters to tune
    """
    def tune_params(self, 
        param_grid = None
    ):
        if param_grid is None:
            param_grid = self.param_grid;
        
        # Add randomness state to param grid
        param_grid["random_state"] = [42];

        # Find best hyperparameters using grid search
        grid_search = GridSearchCV(self.model, param_grid, cv = 5, scoring = "f1", n_jobs = -1); 
        grid_search.fit(self.validate_X, self.validate_y);
        
        self.best_params = grid_search.best_params_;
        self.set_params(self.best_params);

    """
    Set hyperparameters for model
    @param hyperparameters: hyperparameters to set
    """
    def set_params(self, hyperparameters):
        self.model.set_params(**hyperparameters);

        # Retrain model with new hyperparameters
        self.train();

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
        possible_thresholds = np.linspace(0, 1, n + 1);
        # Get confusion matrices for each threshold
        confusion_matrices = [self._calculate_confusion_matrix(threshold) for threshold in possible_thresholds];
        
        # Get optimal threshold index
        optimal_threshold_index = None;

        if optimisation_method == "f1":
            optimal_threshold_index = self._f1_optimisation(confusion_matrices);
        elif optimisation_method == "youden":
            optimal_threshold_index = self._youden_optimisation(confusion_matrices);
        else:
            raise ValueError("Invalid threshold optimisation method");
        
        self.threshold = possible_thresholds[optimal_threshold_index];
        self.confusion_matrix = confusion_matrices[optimal_threshold_index]; 
    """
    Create a confusion matrix using a threshold to determine true/false
    @return: confusion matrix
    """
    def _calculate_confusion_matrix(self, threshold):
        return {
            "true_positive": np.sum((self.pred_y >= threshold) & (self.true_y == 1)),
            "false_positive": np.sum((self.pred_y >= threshold) & (self.true_y == 0)),
            "true_negative": np.sum((self.pred_y < threshold) & (self.true_y == 0)),
            "false_negative": np.sum((self.pred_y < threshold) & (self.true_y == 1))
        };

    """
    Optimise threshold using F1 score
    @param confusion_matrices: confusion matrices for each threshold
    @return: index of optimal threshold
    """
    def _f1_optimisation(self, confusion_matrices):
        # Calculate precisions and recalls for each confusion matrix
        self.precisions = [self._calculate_precision(confusion_matrix) for confusion_matrix in confusion_matrices]; 
        self.recalls = [self._calculate_recall(confusion_matrix) for confusion_matrix in confusion_matrices];
        
        # Calculate f1 scores for each precision and recall
        self.f1s = np.array(
            [self._calculate_f1(precision, recall) for precision, recall in zip(self.precisions, self.recalls)]
        );

        # Find threshold with maximum f1
        max_f1_idx = np.argmax(self.f1s);

        # Store values at optimal threshold 
        self.precision = self.precisions[max_f1_idx];
        self.recall = self.recalls[max_f1_idx];
        self.f1 = self.f1s[max_f1_idx];
        
        # Calculate integral approximation of ROC curve
        self.roc_integral = np.trapz(self.recalls, self.precisions);

        return max_f1_idx;

    """ 
    Optimise threshold using Youden's J statistic
    @param confusion_matrices: confusion matrices for each threshold
    @return: index of optimal threshold
    """
    def _youden_optimisation(self, confusion_matrices):
        # Calculate sensitivities and specificities for each confusion matrix 
        self.sensitivities = [self._calculate_recall(confusion_matrix) for confusion_matrix in self.confusion_matrices];
        self.specificities = [self._calculate_specificity(confusion_matrix) for confusion_matrix in self.confusion_matrices];

        # Calculate Youden's J statistic for each sensitivity and specificity
        self.youdens = np.array(
            [self._calculate_youden(sensitivity, specificity) for sensitivity, specificity in zip(self.sensitivities, self.specificities)]
        );
        
        # Find threshold with maximum Youden's J statistic
        max_youden_idx = np.argmax(self.youdens);
        
        # Store values at optimal threshold
        self.sensitivity = self.sensitivities[max_youden_idx];
        self.specificity = self.specificities[max_youden_idx];
        self.youden = self.youdens[max_youden_idx];

        return max_youden_idx;        

    """
    Calculate F1 score for a given precision and recall 
    @param precision: precision to calculate for
    @param recall: recall to calculate for
    """
    def _calculate_f1(self, precision, recall):
        # Prevent division by zero
        denominator = precision + recall;
        if denominator == 0:
            return 0;

        return 2 * precision * recall / denominator;

    """
    Calculate precision for a given confusion matrix
    @param confusion_matrix: confusion matrix to calculate for
    """
    def _calculate_precision(self, confusion_matrix):
        # Prevent division by zero
        denominator = confusion_matrix["true_positive"] + confusion_matrix["false_positive"];
        if denominator == 0:
            return 0;

        return confusion_matrix["true_positive"] / denominator;

    """
    Calculate recall for a given confusion matrix
    @param confusion_matrix: confusion matrix to calculate for
    """
    def _calculate_recall(self, confusion_matrix):
        # Prevent division by zero
        denominator = confusion_matrix["true_positive"] + confusion_matrix["false_negative"];
        if denominator == 0:
            return 0;

        return confusion_matrix["true_positive"] / denominator;        

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
        # Prevent division by zero
        denominator = confusion_matrix["true_negative"] + confusion_matrix["false_positive"];
        if denominator == 0:
            return 0;

        return confusion_matrix["true_negative"] / denominator;
        
