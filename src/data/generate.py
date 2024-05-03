from typing import Tuple, List

from src.data.datasets import TextData, SentimentData


def generate(templates: List[str], subjects: List[str], adjectives: List[str]) -> Tuple[TextData, SentimentData]:
    raise NotImplementedError
