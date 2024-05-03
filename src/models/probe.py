from src.data.datasets import EncodingData, SentimentData, ProbeDataset


class Probe:
    def train(self, dataset: ProbeDataset):
        raise NotImplementedError

    def predict(self, data: EncodingData) -> SentimentData:
        raise NotImplementedError
