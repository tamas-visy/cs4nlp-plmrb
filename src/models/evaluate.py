import numpy as np

from src.data.datasets import SentimentData


def evaluate(truth: SentimentData, predicted: SentimentData, dummy=False):
    """Calculates metrics based on the real and predicted sentiments"""
    if not dummy:
        raise NotImplementedError

    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    return np.mean((truth - predicted)**2)**0.5
