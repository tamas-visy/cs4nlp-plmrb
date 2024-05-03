from typing import List, Tuple

import numpy as np

from src.data.datasets import TextDataset


def generate(
        templates: List[str], subjects: List[str], adjectives: Tuple[List[str], List[str], List[str]]
) -> TextDataset:
    """A function that generates sentences and ground truth sentiment values from templates with subjects and
    adjectives inserted. Adjectives can be positive, neutral or negative."""
    texts = []
    sentiments = []
    for template in templates:
        for subject in subjects:
            for i, mood in enumerate(adjectives):
                for adjective in mood:
                    sample = template.replace("{X}", subject).replace("{Y}", adjective)
                    gt = np.zeros(3)
                    gt[i] = 1
                    texts.append(sample)
                    sentiments.append(gt)

    return texts, sentiments
