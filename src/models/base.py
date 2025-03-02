import numpy as np

from sklearn.model_selection import GridSearchCV

from abc import ABC, abstractmethod

"""
Base class for a model that can be trained and produces a probability prediction. This class handles fitting, threshold optimisation and evaluation. It also provides public wrappers for prediction.
"""

class BaseClassifier:
    """
    Constructor: set data and default threshold
    """
    def __init__(self, data,
        requested_data_type = "insensitive",
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
        self.test_y = data.test_y;
        
        self._set_thresholds(); 

    """
    Set default and possible thresholds
    @param default_threshold: default threshold to use
    @param possible_thresholds: possible thresholds to consider
    @param n: number of thresholds to consider between 0 and 1
    """
    def _set_thresholds(self,
        default_threshold = 0.5,
        possible_thresholds = None,
        n = 100
    ):
        self.threshold = default_threshold;

        if possible_thresholds is None:
            possible_thresholds = np.linspace(0, 1, n + 1);

        self.possible_thresholds = possible_thresholds;

    """
    Train the model 
    """
    def train(self):
        self.model.fit(self.train_X, self.train_y);
                
        self._set_optimal_threshold();

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
    Abstract method for setting a parameter grid for hyperparameter tuning, to be implemented by parent class (e.g TreeClassifier)
    """
    @abstractmethod
    def _set_param_grid(self, param_grid):
        raise NotImplementedError("Method not implemented");
    
    """
    Tune hyperparameters 
    @param param_grid: possible hyperparameters to tune
    @param cv: number of cross validate folds
    @param scoring: scoring method to use
    @param n_jobs: number of jobs to run in parallel
    """
    def tune(self, 
        param_grid = None,
        cv = 5,
        scoring = "f1",
        n_jobs = -1,
    ):

        if param_grid is None:
            param_grid = self.param_grid;
        
        # Find best hyperparameters using grid search
        grid_search = GridSearchCV(self.model, param_grid, cv = cv, scoring = scoring, n_jobs = n_jobs); 
        grid_search.fit(self.validate_X, self.validate_y);
        
        # Set best hyperparameters
        self.best_params = grid_search.best_params_;

        self.set_params(self.best_params);
    
    """
    Set hyperparameters for model
    @param hyperparameters: hyperparameters to set
    """
    def set_params(self, hyperparameters):
        self.params = hyperparameters;

        self.model.set_params(**self.params);

    """
    Calculate and set optimal threshold value
    @param optimisation_method: method to use for optimisation. f1 is default as data contains class imbalance
    """
    def _set_optimal_threshold(self,
        optimisation_method = "f1",
    ):
        # Store probability predictions for training data 
        self.train_pred_y = self._predict(self.train_X);

        # Get confusion matrices for each threshold
        self.train_confusion_matrices = [self._calculate_confusion_matrix(threshold, self.train_pred_y, self.train_y) for threshold in self.possible_thresholds];        

        # Get optimal threshold index
        if optimisation_method == "f1":
            self._f1_optimisation(self.train_confusion_matrices);
        elif optimisation_method == "youden":
            self._youden_optimisation(self.train_confusion_matrices);
        else:
            raise ValueError("Invalid threshold optimisation method");
        
        self.threshold = self.possible_thresholds[self.optimal_threshold_index];
    
    """ 
    Create a confusion matrix using a threshold to determine true/false
    @return: confusion matrix
    """
    def _calculate_confusion_matrix(self, threshold, pred_y, true_y):
        return {
            "true_positive": np.sum((pred_y >= threshold) & (true_y == 1)),
            "false_positive": np.sum((pred_y >= threshold) & (true_y == 0)),
            "true_negative": np.sum((pred_y < threshold) & (true_y == 0)),
            "false_negative": np.sum((pred_y < threshold) & (true_y == 1))
        };
  
    """
    Optimise threshold using F1 score
    @param confusion_matrices: confusion matrices for each threshold
    """
    def _f1_optimisation(self, confusion_matrices):
        # Calculate precisions and recalls for each confusion matrix
        self.validate_precisions = [self._calculate_precision(confusion_matrix) for confusion_matrix in confusion_matrices]; 
        self.validate_recalls = [self._calculate_recall(confusion_matrix) for confusion_matrix in confusion_matrices];
        
        # Calculate f1 scores for each precision and recall
        self.validate_f1s = np.array(
            [self._calculate_f1(precision, recall) for precision, recall in zip(self.validate_precisions, self.validate_recalls)]
        );

        # Find threshold with maximum f1
        self.optimal_threshold_index = np.argmax(self.validate_f1s);

        # Store evaluation metrics at optimal threshold 
        self.validate_precision = self.validate_precisions[self.optimal_threshold_index]; 
        self.validate_recall = self.validate_recalls[self.optimal_threshold_index];
        self.validate_f1 = self.validate_f1s[self.optimal_threshold_index];
        self.validate_roc_integral = np.trapz(self.validate_recalls, self.validate_precisions);

    """ 
    Optimise threshold using Youden's J statistic
    @param confusion_matrices: confusion matrices for each threshold
    """
    def _youden_optimisation(self, confusion_matrices):
        # Calculate sensitivities and specificities for each confusion matrix
        self.validate_sensitivities = [self._calculate_recall(confusion_matrix) for confusion_matrix in confusion_matrices]; 
        self.validate_specificities = [self._calculate_specificity(confusion_matrix) for confusion_matrix in confusion_matrices];
        
        # Calculate Youden's J statistic for each sensitivity and specificity
        self.validate_youdens = np.array(
        
                [self._calculate_youden(sensitivity, specificity) for sensitivity, specificity in zip(self.sensitivities, self.specificities)]
        );

        # Find threshold with maximum Youden's J statistic
        self.optimal_threshold_index = np.argmax(self.validate_youdens);

        # Store evaluation metrics at optimal threshold
        self.validate_sensitivity = self.validate_sensitivities[self.optimal_threshold_index]; 
        self.validate_specificity = self.validate_specificities[self.optimal_threshold_index];
        self.validate_youden = self.validate_youdens[self.optimal_threshold_index];

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
     
    """
    Evaluate model using test data
    """
    def evaluate(self,
        metric = "f1"
    ):
        self.test_pred_y = self._predict(self.test_X);
        
        # Calculate confusion matrices based on test data for each threshold
        self.confusion_matrices = [self._calculate_confusion_matrix(threshold, self.test_pred_y, self.test_y) for threshold in self.possible_thresholds];        
        self.confusion_matrix = self.confusion_matrices[self.optimal_threshold_index];

        # Store evaluation metrics
        if metric == "f1":
            self._evaluate_f1();
        elif metric == "youden":
            self._evaluate_youden();
        else:
            raise ValueError("Invalid evaluation metric");
    
    """
    Store f1 metrics based on test data
    """
    def _evaluate_f1(self):
        # Calculate precisions and recalls for all possible thresholds
        self.precisions = [self._calculate_precision(confusion_matrix) for confusion_matrix in self.confusion_matrices]; 
        self.recalls = [self._calculate_recall(confusion_matrix) for confusion_matrix in self.confusion_matrices];

        # Calculate f1s and integral of ROC curve
        self.f1s = [self._calculate_f1(precision, recall) for precision, recall in zip(self.precisions, self.recalls)];
        self.roc_integral = np.trapz(self.recalls, self.precisions);
        
        # Calculate f1 for optimal threshold
        self.precision = self.precisions[self.optimal_threshold_index];
        self.recall = self.recalls[self.optimal_threshold_index];
        self.f1 = self.f1s[self.optimal_threshold_index];
    
    """
    Store Youden metrics based on test data
    """
    def _evaluate_youden(self): 
        # Calculate sensitivities and specificities for all possible thresholds
        self.sensitivities = [self._calculate_recall(confusion_matrix) for confusion_matrix in self.confusion_matrices]; 
        self.specificities = [self._calculate_specificity(confusion_matrix) for confusion_matrix in self.confusion_matrices];

        # Calculate Youden's J statistic and integral of ROC curve 
        self.youdens = [self._calculate_youden(sensitivity, specificity) for sensitivity, specificity in zip(self.sensitivities, self.specificities)];
        self.integral_roc = np.trapz(self.sensitivities, self.specificities);

        # Calculate Youden's J statistic for optimal threshold
        self.sensitivity = self.sensitivities[self.optimal_threshold_index];
        self.specificity = self.specificities[self.optimal_threshold_index];
        self.youden = self.youdens[self.optimal_threshold_index];

    
       
