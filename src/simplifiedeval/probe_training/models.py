import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
