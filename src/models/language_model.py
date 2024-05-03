from src.data.datasets import LanguageData, EncodingData


class LanguageModel:
    def encode(self, x: LanguageData) -> EncodingData:
        """Returns the encodings of x"""
        raise NotImplementedError
