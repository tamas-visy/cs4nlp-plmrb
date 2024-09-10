import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from datasets import Dataset

# Paths and constants
data_path = "/content/drive/MyDrive/cs4nlp-plmrb-main/data/processed"
out_path = "/content/drive/MyDrive/outputs_Hf_shuffle"
train_hash = "499193892"
test_hashes = ["336359147", "315634198"]
layers = ["initial", "middle", "final"]

# Load and extract labels
train_labels_df = pd.read_csv("/content/drive/MyDrive/cs4nlp-plmrb-main/data/processed/train_dataset_processed.csv")
dataset = Dataset.from_pandas(train_labels_df)
dataset = dataset.shuffle(seed=42)
labels = dataset['label']


def process_and_evaluate(layer, model_name, model, train_data, train_labels, val_data, val_labels, test_data_1,
                         test_data_2):
    model.fit(train_data, train_labels)
    val_preds = model.predict(val_data)
    val_accuracy = accuracy_score(val_labels, val_preds)
    train_preds = model.predict(train_data)
    train_accuracy = accuracy_score(train_labels, train_preds)

    test_data_1_pred_labels = model.predict(test_data_1)
    test_data_1_pred_pos_probs = model.predict_proba(test_data_1)[:, 1]

    test_data_2_pred_labels = model.predict(test_data_2)
    test_data_2_pred_pos_probs = model.predict_proba(test_data_2)[:, 1]

    test_preds_labels = pd.DataFrame({
        "pos_prob1": test_data_1_pred_pos_probs,
        "pos_prob2": test_data_2_pred_pos_probs
    })
    # During visualization or wtv, map negative to -1 and positive to +1

    return train_accuracy, val_accuracy, test_preds_labels


def main():
    """
    The script starts by importing necessary libraries for data manipulation and machine learning, including `numpy`,
    `pandas`, `sklearn`, and `datasets`. Paths and constants are set up for data and output directories,
    dataset identifiers, and layer names. Training labels are loaded from a CSV file into a pandas DataFrame,
    converted into a Hugging Face `Dataset`, and shuffled. Labels are extracted from the dataset. The
    `process_and_evaluate` function is defined to train a given model using training data, evaluate it on validation
    data, and generate predictions on two test datasets, returning accuracy scores and a DataFrame of predicted
    probabilities. The main loop iterates over directories in the data path, focusing only on the `GloveLanguageModel`
    folder. For each specified layer, the script loads training and test data from `.npy` files, and splits the training
    data into training and validation sets. A dictionary of models is created, including Random Forest and Gaussian Naive
    Bayes. For each model, the `process_and_evaluate` function is called to train and evaluate the model. Results,
    including test predictions and accuracies, are saved to the specified output directory. Additionally, results for the
    "middle" and "final" layers are saved using the same trained model, ensuring consistency across layers.
    """
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
                    # Train data has 0-1 labels
                    test_data_file_1 = os.path.join(lm_path, f"{layer}_{test_hashes[0]}.npy")
                    test_data_file_2 = os.path.join(lm_path, f"{layer}_{test_hashes[1]}.npy")
                    # Test data has -1, 1 labels

                    train_data = np.load(train_file)
                    test_data_1 = np.load(test_data_file_1)
                    test_data_2 = np.load(test_data_file_2)

                    train_data, val_data, train_labels, val_labels = train_test_split(train_data, labels, test_size=0.2,
                                                                                      random_state=42)

                    models = {
                        # "logistic_regression": LogisticRegression(max_iter=1000),
                        # "svm": SVC(probability=True),
                        "random_forest": RandomForestClassifier(n_estimators=50, random_state=42),
                        # "k-nn": KNeighborsClassifier(n_neighbors=100),
                        "gaussian_nb": GaussianNB()
                        # "mlp": MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
                    }
                    for model_name, model in models.items():
                        print("\t\tTraining", model_name, "model")
                        train_accuracy, val_accuracy, test_preds_probs = process_and_evaluate(layer, model_name, model,
                                                                                              train_data, train_labels,
                                                                                              val_data, val_labels,
                                                                                              test_data_1, test_data_2)
                        print("\t\tTraining for", model_name, "model complete!")

                        output_dir = os.path.join(out_lm_path, model_name, layer)
                        os.makedirs(output_dir, exist_ok=True)

                        test_preds_probs.to_csv(os.path.join(output_dir, "test_preds_probs.csv"), index=False)

                        with open(os.path.join(output_dir, "accuracy.txt"), 'w') as f:
                            f.write(f"Training Accuracy: {train_accuracy}\n")
                            f.write(f"Validation Accuracy: {val_accuracy}\n")

                        layer_alt = ['middle', 'final']
                        for layer_mf in layer_alt:
                            alt_output_dir = os.path.join(out_lm_path, model_name, layer_mf)
                            os.makedirs(alt_output_dir, exist_ok=True)

                            test_preds_probs.to_csv(os.path.join(alt_output_dir, "test_preds_probs.csv"), index=False)

                            with open(os.path.join(alt_output_dir, "accuracy.txt"), 'w') as f:
                                f.write(f"Training Accuracy: {train_accuracy}\n")
                                f.write(f"Validation Accuracy: {val_accuracy}\n")


if __name__ == '__main__':
    main()
