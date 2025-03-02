from src.data_main import Data
from src.models_main import Models

"""
A main script to run the entire pipeline, returns the same result as the Jupiter notebook.
"""

# Data Preprocessing
print("Initialising data...");
data = Data();

print("Data initialisation complete. Preprocessing data...");
data.preprocess();

# Modelling and Evaluation
print("Data preprocessing complete. Initialising models...");
models = Models(data);

print("Models initialisation complete. Tuning models...");
models.tune(load_path = "src/hyperparameters.json");

print("Model tuning complete. Training models...");
models.train();

print("Model training complete. Evaluating models...");
models.evaluate();

breakpoint();
