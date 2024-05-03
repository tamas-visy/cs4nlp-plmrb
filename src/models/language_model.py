from src.data.datasets import TextData, EncodingData


class LanguageModel:
    def encode(self, x: TextData) -> EncodingData:
        """Returns the encodings of x"""
        raise NotImplementedError
