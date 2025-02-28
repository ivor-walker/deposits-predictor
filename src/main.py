from data import Data
from models import Models

# Data Preprocessing
data = Data();
data.preprocessing();

# Modelling and Evaluation
models = Models(data);
models.train();

models.get_confusion_matrices();
models.get_f1_scores();
