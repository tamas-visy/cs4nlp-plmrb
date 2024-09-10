import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from src.simplifiedeval.probe_training.models import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def process_and_evaluate_cpu(layer, model_name, model, train_data, train_labels, val_data, val_labels, test_data_1,
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


def process_and_evaluate_gpu(layer, model_name, model, train_data, train_labels, val_data, val_labels, test_data_1,
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
