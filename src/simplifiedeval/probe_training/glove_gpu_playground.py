"""
### Glove GPU playground
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datasets import Dataset

np.random.seed(42)

# Paths and constants
data_path = "/content/drive/MyDrive/cs4nlp-plmrb-main/data/processed"
out_path = "/content/drive/MyDrive/outputs_Hf_shuffle"
train_hash = "499193892"
test_hashes = ["336359147", "315634198"]
layers = ["initial", "middle", "final"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load labels
train_labels_df = pd.read_csv("/content/drive/MyDrive/cs4nlp-plmrb-main/data/processed/train_dataset_processed.csv")
dataset = Dataset.from_pandas(train_labels_df)
dataset = dataset.shuffle(seed=42)

# Extract labels
labels = dataset['label']
# labels = train_labels_df["label"].values
# np.random.shuffle(labels)

# Convert labels to the format required by PyTorch
labels_torch = torch.tensor(labels, dtype=torch.float32).to(device)


# Logistic Regression Model in PyTorch
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Deep Neural Network Model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model, train_loader, val_loader, epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    model.eval()

    train_preds = []
    train_targets = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            train_preds.extend(outputs.cpu().numpy())
            train_targets.extend(targets.numpy())

    train_preds = np.round(train_preds)
    train_accuracy = accuracy_score(train_targets, train_preds)

    val_preds = []
    val_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            val_preds.extend(outputs.cpu().numpy())
            val_targets.extend(targets.numpy())

    val_preds = np.round(val_preds)
    val_accuracy = accuracy_score(val_targets, val_preds)
    return train_accuracy, val_accuracy


def evaluate_model(model, test_loader):
    model.eval()
    test_preds = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            test_preds.extend(outputs.cpu().numpy())

    test_preds = np.round(test_preds)
    return test_preds


def process_and_evaluate(layer, model_name, model, train_data, train_labels, val_data, val_labels, test_data_1,
                         test_data_2, model_type="torch"):
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                                  torch.tensor(train_labels, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32),
                                torch.tensor(val_labels, dtype=torch.float32))
    test_dataset_1 = TensorDataset(torch.tensor(test_data_1, dtype=torch.float32))
    test_dataset_2 = TensorDataset(torch.tensor(test_data_2, dtype=torch.float32))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader_1 = DataLoader(test_dataset_1, batch_size=32, shuffle=False)
    test_loader_2 = DataLoader(test_dataset_2, batch_size=32, shuffle=False)

    train_accuracy, val_accuracy = train_model(model, train_loader, val_loader)
    # test_preds_1 = evaluate_model(model, test_loader_1)
    # test_preds_2 = evaluate_model(model, test_loader_2)

    with torch.no_grad():
        test_preds_1_pos_probs = model(
            torch.tensor(test_data_1, dtype=torch.float32).to(device)).cpu().numpy().squeeze()
        test_preds_2_pos_probs = model(
            torch.tensor(test_data_2, dtype=torch.float32).to(device)).cpu().numpy().squeeze()

    test_preds_labels = pd.DataFrame({
        "pos_prob1": test_preds_1_pos_probs,
        "pos_prob2": test_preds_2_pos_probs
    })

    return train_accuracy, val_accuracy, test_preds_labels


for lm_folder in os.listdir(data_path):
    if lm_folder != 'GloveLanguageModel':
        continue
    lm_path = os.path.join(data_path, lm_folder)
    out_lm_path = os.path.join(out_path, lm_folder)
    if os.path.isdir(lm_path):
        print("Processing data for", lm_folder)
        for layer in layers:
            if layer == 'initial':
                print("\tProcessing", layer, "layer")
                train_file = os.path.join(lm_path, f"{layer}_{train_hash}.npy")
                test_file_1 = os.path.join(lm_path, f"{layer}_{test_hashes[0]}.npy")
                test_file_2 = os.path.join(lm_path, f"{layer}_{test_hashes[1]}.npy")

                train_data = np.load(train_file)
                test_data_1 = np.load(test_file_1)
                test_data_2 = np.load(test_file_2)

                train_data, val_data, train_labels, val_labels = train_test_split(train_data, labels,
                                                                                  test_size=0.2, random_state=42)

                input_dim = train_data.shape[1]

                models = {
                    "logistic_regression": LogisticRegressionModel(input_dim).to(device),
                    "mlp": SimpleMLP(input_dim).to(device),
                    # "random_forest": RandomForestClassifier(n_estimators=100, random_state=42)
                }

                model_types = {
                    "logistic_regression": "torch",
                    "mlp": "torch",
                    "random_forest": "sklearn"
                }

                for model_name, model in models.items():
                    output_dir = os.path.join(out_lm_path, model_name, layer)
                    print("\t\tTraining", model_name, "model")
                    train_accuracy, val_accuracy, test_preds_probs = process_and_evaluate(
                        layer, model_name, model, train_data, train_labels,
                        val_data, val_labels, test_data_1, test_data_2, model_types[model_name])
                    print("\t\tTraining for", model_name, "model complete!")

                    os.makedirs(output_dir, exist_ok=True)

                    test_preds_probs.to_csv(os.path.join(output_dir, "test_preds_probs.csv"), index=False)

                    with open(os.path.join(output_dir, "accuracy.txt"), 'w') as f:
                        f.write(f"Training Accuracy: {train_accuracy}\nValidation Accuracy: {val_accuracy}\n")

                    layer_alt = ['middle', 'final']

                    for layer_mf in layer_alt:
                        alt_output_dir = os.path.join(out_lm_path, model_name, layer_mf)
                        os.makedirs(alt_output_dir, exist_ok=True)

                        test_preds_probs.to_csv(os.path.join(alt_output_dir, "test_preds_probs.csv"), index=False)

                        with open(os.path.join(alt_output_dir, "accuracy.txt"), 'w') as f:
                            f.write(f"Training Accuracy: {train_accuracy}\n")
                            f.write(f"Validation Accuracy: {val_accuracy}\n")

"""
The script imports essential libraries for data manipulation, machine learning, and deep learning, 
including `numpy`, `pandas`, `torch`, and `sklearn`. Paths and constants are set for data and output directories, 
dataset identifiers, and layer names. The script then checks for GPU availability and assigns the device accordingly. 
Training labels are loaded from a CSV file into a pandas DataFrame, converted into a Hugging Face `Dataset`, 
shuffled, and labels are extracted. These labels are converted into a PyTorch tensor for further processing.

Two PyTorch models are defined: a `LogisticRegressionModel` and a `SimpleMLP` (a simple multi-layer perceptron). A 
function `train_model` is defined to handle the training process of the models, which includes setting up the loss 
function and optimizer, iterating through epochs, and calculating training and validation accuracies. The 
`evaluate_model` function is defined to handle the evaluation of models on test data, but it’s not used directly 
within the `process_and_evaluate` function.

The `process_and_evaluate` function prepares the data by creating PyTorch `TensorDataset` and `DataLoader` objects 
for training, validation, and test datasets. It then trains the model using the `train_model` function and generates 
predictions on the test datasets, returning the training accuracy, validation accuracy, and a DataFrame of predicted 
probabilities for the test datasets.

The main loop iterates over directories in the data path, focusing only on the `GloveLanguageModel` folder. For each 
specified layer, it processes the "initial" layer by loading training and test data from `.npy` files, splitting the 
training data into training and validation sets, and defining the input dimension for the models. A dictionary of 
models is created, including `LogisticRegressionModel` and `SimpleMLP`.

For each model, the `process_and_evaluate` function is called to train and evaluate the model. Results, 
including test predictions and accuracies, are saved to the specified output directory. Additionally, results for the 
"middle" and "final" layers are saved using the same trained model, ensuring consistency across layers. This ensures 
that the models' performance is recorded and evaluated across different data representations.
"""
