import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset

from src.simplifiedeval.probe_training.models import SimpleMLP
from src.simplifiedeval.probe_training.utils import process_and_evaluate_gpu

np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(data_path, out_path, train_hash, test_hashes, layers):
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
    # Load and extract labels
    train_labels_df = pd.read_csv("/content/drive/MyDrive/cs4nlp-plmrb-main/data/processed/train_dataset_processed.csv")
    dataset = Dataset.from_pandas(train_labels_df)
    dataset = dataset.shuffle(seed=42)
    labels = dataset['label']

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
                    train_accuracy, val_accuracy, test_preds_probs = process_and_evaluate_gpu(
                        layer, model_name, model, train_data, train_labels,
                        val_data, val_labels, test_data_1, test_data_2, model_types[model_name])
                    print("Training for", model_name, "model complete!")

                    os.makedirs(output_dir, exist_ok=True)

                    test_preds_probs.to_csv(os.path.join(output_dir, "test_preds_probs.csv"), index=False)

                    with open(os.path.join(output_dir, "accuracy.txt"), 'w') as f:
                        f.write(f"Training Accuracy: {train_accuracy}\nValidation Accuracy: {val_accuracy}\n")


if __name__ == '__main__':
    main()
