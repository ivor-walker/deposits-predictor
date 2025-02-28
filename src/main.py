from data import Data
from models_main import Models

import pdb

# Data Preprocessing
data = Data();
data.preprocess();

# Modelling and Evaluation
models = Models(data);

models.train();

breakpoint();

models.get_confusion_matrices();
models.get_f1_scores();
