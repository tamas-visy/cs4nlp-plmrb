from typing import List, Tuple

from datasets import Dataset

from src.data.datatypes import TextDataset, SubjectDataset


def generate(
        templates: List[str], subjects: List[str], adjectives: Tuple[List[str], List[str], List[str]]
) -> TextDataset | SubjectDataset:
    """A function that generates sentences and ground truth sentiment values from templates with subjects and
    adjectives inserted. Adjectives can be positive, neutral or negative."""
    subjects_ = []
    texts = []
    sentiments = []
    for template in templates:
        for subject in subjects:
            for i, mood in enumerate(adjectives):
                for adjective in mood:
                    sample = template.replace("{X}", subject).replace("{Y}", adjective)

                    # Create label
                    # TODO label must be related to dataset_1's label
                    #   e.g. gt = np.zeros(3); gt[i] = 1 for dummy dataset
                    match i:
                        case 0:
                            gt = 0.0
                        case 1:
                            gt = 0.5
                        case 2:
                            gt = 1.0
                        case _:
                            raise ValueError

                    subjects_.append(subject)
                    texts.append(sample)
                    sentiments.append(gt)

    return Dataset.from_dict(dict(subject=subjects_, input=texts, label=sentiments))
