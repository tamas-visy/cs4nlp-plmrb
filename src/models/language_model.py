import numpy as np

from src.data.datasets import TextData, EncodingData


class LanguageModel:
    def encode(self, texts: TextData) -> EncodingData:
        """Returns the encodings of texts"""
        raise NotImplementedError


class DummyLanguageModel(LanguageModel):
    def __init__(self):
        self.embedding_size = 16

    def encode(self, texts: TextData) -> EncodingData:
        return [np.random.random(self.embedding_size) for _ in texts]
