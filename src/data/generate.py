from typing import List, Tuple, Dict

from datasets import Dataset

from src.data.datatypes import TextDataset, GroupsDataset


def generate(
        templates: List[str],
        groups: Dict[str, List[str]],
        adjectives: Tuple[List[str], List[str], List[str]]
) -> TextDataset | GroupsDataset:
    """A function that generates sentences and ground truth sentiment values from templates with groups and
    adjectives inserted. Adjectives can be positive, neutral or negative."""
    groups_ = []
    texts = []
    sentiments = []
    for template in templates:
        for group_name, group in groups.items():
            for subject in group:
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

                        groups_.append(group_name)
                        texts.append(sample)
                        sentiments.append(gt)

    return Dataset.from_dict(dict(group=groups_, input=texts, label=sentiments))
