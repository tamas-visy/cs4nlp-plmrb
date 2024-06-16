import logging

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression
from src.data.datatypes import EncodingData, SentimentData, EncodingDataset
from src.data.iohandler import IOHandler
from src.utils.misc import stable_hash

logger = logging.getLogger(__name__)


class Probe:
    def __init__(self):
        self._cache = IOHandler.load_cache_of(self)

    def train(self, dataset: EncodingDataset):
        logger.debug(f"{self.__class__.__name__} is training on {len(dataset)} sentences")
        self._train(dataset)

    def predict(self, data: EncodingData, sentences=None) -> SentimentData:
        # sentences is only used for caching
        logger.debug(f"{self.__class__.__name__} is predicting {len(data)} sentences")

        # cache_key = (str(stable_hash('='.join(sentences)))[:9], ) if sentences is not None else None
        # if cache_key in self._cache:
        #     logger.debug(f"Found {cache_key} in cache, using that instead of predicting again")
        #     return self._cache[cache_key]

        out = self._predict(data)
        # if cache_key is not None:
        #     IOHandler.save_cache_of(self, out, *cache_key)

        return out

    def _train(self, dataset: EncodingDataset):
        raise NotImplementedError

    def _predict(self, data: EncodingData) -> SentimentData:
        raise NotImplementedError


class SKLearnProbe(Probe):
    """A base class that can be used by scikit-learn predictors"""

    def __init__(self, model=None):
        super().__init__()
        if model is None:
            raise ValueError
        self._model = model

    def _train(self, dataset: EncodingDataset):
        self._model.fit(dataset["input"], dataset["label"])

    def _predict(self, data: EncodingData) -> SentimentData:
        return self._model.predict(data)


class LinearProbe(SKLearnProbe):
    """A probe performing linear regression"""

    def __init__(self):
        super().__init__(LogisticRegression())


class SVMProbe(SKLearnProbe):
    """A probe based on Support Vector Machines. Can perform single variable regression"""

    def __init__(self):
        super().__init__(SVR())
        # super().__init__(SVC())  # Classification version

    def _train(self, dataset: EncodingDataset):
        if len(dataset) > 10_000:
            logger.warning(f"{self.__class__.__name__}'s train time is more than quadratic. This could be slow")
        super()._train(dataset)


class MLPProbe(SKLearnProbe):
    """A probe using a Multilayer perceptron with 2 hidden layers and early stopping."""

    def __init__(self):
        super().__init__(MLPRegressor(hidden_layer_sizes=[128, 128], early_stopping=True))
        # super().__init__(MLPClassifier(hidden_layer_sizes=[128, 128], early_stopping=True))   # Classification version
