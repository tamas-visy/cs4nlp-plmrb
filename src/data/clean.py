from src.data.datasets import TextDataset


def clean_dataset(dataset: TextDataset, dummy=False) -> TextDataset:
    """Cleans the dataset using cleaning measures such as filtering via string matching or Named Entity Recognition."""
    # Can use NLTK or spaCy
    if not dummy:
        raise NotImplementedError
    # Dummy version drops first row
    # TODO use types that make this easier
    return dataset[0][1:], dataset[1][1:]
