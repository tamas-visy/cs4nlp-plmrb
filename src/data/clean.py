from src.data.datatypes import TextDataset


def clean_dataset(dataset: TextDataset, dummy=False) -> TextDataset:
    """Cleans the dataset using cleaning measures such as filtering via string matching or Named Entity Recognition."""
    # Can use NLTK or spaCy
    if not dummy:
        raise NotImplementedError
    # Dummy version drops rows with "London" in them
    dataset = dataset.filter(lambda row: "London".lower() not in row['input'])
    return dataset
