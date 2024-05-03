from src.data.datasets import EncodingData, SentimentData


class Probe:
    def train(self, dataset: EncodingData):
        raise NotImplementedError

    def predict(self, dataset: EncodingData) -> SentimentData:
        raise NotImplementedError
