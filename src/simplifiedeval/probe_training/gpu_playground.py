import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
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

# Load and extract labels
train_labels_df = pd.read_csv("/content/drive/MyDrive/cs4nlp-plmrb-main/data/processed/train_dataset_processed.csv")
dataset = Dataset.from_pandas(train_labels_df)
dataset = dataset.shuffle(seed=42)
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


def main():
    """
    This script begins by importing necessary libraries for data manipulation, machine learning, and neural network
    operations, including `numpy`, `pandas`, `torch`, and `sklearn`, along with setting a random seed for
    reproducibility. It defines paths and constants for data and output directories, dataset identifiers,
    and layer names, and determines whether to use a GPU or CPU. The training labels are loaded from a CSV file into a
    pandas DataFrame, converted into a Hugging Face `Dataset`, and shuffled. The labels are extracted and converted to a
    PyTorch tensor for use in training. Two PyTorch neural network models are defined: a logistic regression model,
    and a simple multi-layer perceptron (MLP), each with specified architectures and forward pass methods. The
    `train_model` function trains a given model using training data and evaluates it on validation data,
    returning accuracy scores. The `evaluate_model` function generates predictions on test data. The
    `process_and_evaluate` function processes the data, trains the model, and evaluates it on training, validation,
    and two test datasets (with and without mask), returning accuracies and a DataFrame of predicted probabilities. The
    main loop iterates over directories in the data path, processes data for each specified layer, loads the training and
    test data, splits the training data into training and validation sets, and initializes a dictionary of models to be
    trained. For each model, the `process_and_evaluate` function is called to train and evaluate the model, with results
    saved to the specified output directory.
    """
    for lm_folder in os.listdir(data_path):
        if lm_folder == 'GloveLanguageModel':
            continue
        lm_path = os.path.join(data_path, lm_folder)
        out_lm_path = os.path.join(out_path, lm_folder)
        if os.path.isdir(lm_path):
            print("Processing data for", lm_folder)
            for layer in layers:
                print("Processing", layer, "layer")
                train_file = os.path.join(lm_path, f"mean_{layer}_{train_hash}.npy")
                test_file_1 = os.path.join(lm_path, f"mean_{layer}_{test_hashes[0]}.npy")
                test_file_2 = os.path.join(lm_path, f"mean_{layer}_{test_hashes[1]}.npy")

                train_data = np.load(train_file)
                test_data_1 = np.load(test_file_1)
                test_data_2 = np.load(test_file_2)

                train_data, val_data, train_labels, val_labels = train_test_split(train_data, labels,
                                                                                  test_size=0.2, random_state=42)

                input_dim = train_data.shape[1]

                models = {
                    # "logistic_regression": LogisticRegressionModel(input_dim).to(device),
                    "mlp": SimpleMLP(input_dim).to(device),
                }

                model_types = {
                    "logistic_regression": "torch",
                    "mlp": "torch",
                    "random_forest": "sklearn",
                    "LSTM": "torch"
                }

                for model_name, model in models.items():
                    output_dir = os.path.join(out_lm_path, model_name, layer)
                    if os.path.isdir(output_dir):
                        continue
                    print("Training", model_name, "model")
                    train_accuracy, val_accuracy, test_preds_probs = process_and_evaluate(
                        layer, model_name, model, train_data, train_labels,
                        val_data, val_labels, test_data_1, test_data_2, model_types[model_name])
                    print("Training for", model_name, "model complete!")

                    os.makedirs(output_dir, exist_ok=True)

                    test_preds_probs.to_csv(os.path.join(output_dir, "test_preds_probs.csv"), index=False)

                    with open(os.path.join(output_dir, "accuracy.txt"), 'w') as f:
                        f.write(f"Training Accuracy: {train_accuracy}\nValidation Accuracy: {val_accuracy}\n")


if __name__ == '__main__':
    main()
