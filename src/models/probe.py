import logging

from sklearn.linear_model import LinearRegression
from src.data.datatypes import EncodingData, SentimentData, ProbeDataset

logger = logging.getLogger(__name__)


class Probe:
    def train(self, dataset: ProbeDataset):
        logger.debug(f"{self.__class__.__name__} is training on {len(dataset)} sentences")
        self._train(dataset)

    def predict(self, data: EncodingData) -> SentimentData:
        logger.debug(f"{self.__class__.__name__} is predicting {len(data)} sentences")
        return self._predict(data)

    def _train(self, dataset: ProbeDataset):
        raise NotImplementedError

    def _predict(self, data: EncodingData) -> SentimentData:
        raise NotImplementedError


class LinearProbe(Probe):
    """A probe performing linear regression"""

    def __init__(self):
        self._model = LinearRegression()

    def _train(self, dataset: ProbeDataset):
        self._model.fit(dataset["input"], dataset["label"])

    def _predict(self, data: EncodingData) -> SentimentData:
        return self._model.predict(data)
