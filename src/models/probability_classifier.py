"""
probability_classifier: Extension of BaseClassifier to handle probability-based classification. Contains threshold optimisation methods.
"""
class ProbabilityClassifier(BaseClassifier):
    
    """
    Constructor: call BaseClassifier constructor and set a default threshold
    """
    def __init__(self,
        data,
        default_threshold = 0.5
    ):
        super().__init__(data);
        self.threshold = default_threshold;
    
    """
    Override base class training to include calculate optimal threshold
    """
    def train(self):
        super().train();
        self._calculate_optimal_threshold();
    
    """
    Override base class predict to use threshold
    @param test_X: data to generate predictions for
    """
    def predict(self, test_X):
        pred_y = super().predict(test_X); 
        return pred_y > self.threshold;

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
        
        # Get threshold that maximises f1
        if optimisation_method == "f1":
            _calculate_precisions_recalls(possible_thresholds);
            _calculate_f1s();
            self.threshold = possible_thresholds[np.argmax(self.f1s)];
        
        # Get threshold that maximises Youden's J
        elif optimisation_method == "youden":
            _calculate_sensitivities_specificities(possible_thresholds);
            _calculate_youdens();
            self.threshold = possible_thresholds[np.argmax(self.youdens)];
        
        else:
            raise ValueError("Invalid optimisation method");

    """
    Override base class confusion matrix to use a threshold to determine true/false
    @return: confusion matrix
    """
    def calculate_confusion_matrix(self,
        threshold = None 
    ):
        if threshold is None:
            threshold = self.threshold;

        return {
            "false_negative": np.sum(self.pred_y <= threshold) & (self.true_y == 1));
            "false_positive": np.sum(self.pred_y > threshold) & (self.true_y == 0));
            "true_negative": np.sum(self.pred_y <= threshold) & (self.true_y == 0));
            "true_positive": np.sum(self.pred_y > threshold) & (self.true_y == 1));
        };

    """
    Calculate F1 scores for all precisions and recalls
    """
    def _calculate_f1s(self):
        # Calculate F1 scores of all precisions and recalls
        self.f1s = [self._calculate_f1(precision, recall) for precision, recall in zip(self.precisions, self.recalls)]; 

    """
    Calculate precision and recall for given thresholds
    @param possible_thresholds: array of possible thresholds
    """
    def _calculate_precisions_recalls(self, possible_thresholds):
        self.precisions = [];
        self.recalls = [];

        for threshold in possible_thresholds: 
            confusion_matrix = self.calculate_confusion_matrix(threshold = threshold);
            self.precisions.append(self._calculate_precision(confusion_matrix));
            self.recalls.append(self._calculate_recalls(confusion_matrix));
    
    """
    Calculate Youden's J statistic for all given sensitivities and specificities
    """
    def _calculate_youdens(self):
        self.youdens = [self._calculate_youden(sensitivity, specificity) for sensitivity, specificity in zip(self.sensitivities, self.specificities)];

    """
    Calculate sensitivity and specificity for all given thresholds
    @param possible_thresholds: array of possible thresholds
    """
    def _calculate_sensitivities_specificities(self, possible_thresholds):
        self.sensitivities = [];
        self.specificities = [];

        for threshold in possible_thresholds: 
            confusion_matrix = self.calculate_confusion_matrix(threshold = threshold);
            self.sensitivities.append(self._calculate_recall(confusion_matrix));
            self.specificities.append(self._calculate_specificity(confusion_matrix));
