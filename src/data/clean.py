import logging

from src.data.datatypes import TextDataset
from src.data.dropping import process_sentence

logger = logging.getLogger(__name__)


def clean_dataset(dataset: TextDataset, dummy=False) -> TextDataset:
    """Cleans the dataset using cleaning measures such as filtering via string matching or Named Entity Recognition."""
    # Can use NLTK or spaCy
    if not dummy:
        dataset = dataset.map(lambda row: {"input": process_sentence(row["input"]), "label": row["label"]})
    else:
        # Dummy version drops rows with "London" in them
        dataset = dataset.filter(lambda row: "London".lower() not in row['input'])
        logger.warning("This is a dummy implementation")
    return dataset
