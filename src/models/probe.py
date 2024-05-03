from sklearn.linear_model import LinearRegression
from src.data.datasets import EncodingData, SentimentData, ProbeDataset


class Probe:
    def train(self, dataset: ProbeDataset):
        raise NotImplementedError

    def predict(self, data: EncodingData) -> SentimentData:
        raise NotImplementedError


class DummyProbe(Probe):
    def __init__(self):
        self._model = LinearRegression()

    def train(self, dataset: ProbeDataset):
        self._model.fit(dataset[0], dataset[1])

    def predict(self, data: EncodingData) -> SentimentData:
        return self._model.predict(data)
