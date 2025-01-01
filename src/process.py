import logging
from typing import Literal, Type

import pandas as pd
from datasets import Dataset

from src.data.datatypes import TextDataset, EncodingData, EncodingDataset
from src.models.evaluate import evaluate
from src.models.language_model import TransformerModel
from src.models.probe import Probe

logger = logging.getLogger(__name__)


def complete(
        lm: TransformerModel,
        result_type: int | Literal["initial", "final", "middle"],
        probe_factory: Type[Probe],
        dataset_1: TextDataset,
        dataset_2: EncodingDataset,
        only_generate_encodings=False,
) -> pd.DataFrame:
    probe = probe_factory()
    
    encodings: EncodingData = lm.encode(dataset_1["input"], result_type=result_type)

    if not only_generate_encodings:
        probe.train(
            dataset=Dataset.from_dict(dict(input=encodings, label=dataset_1["label"]))
        )
        logger.info(f"Trained probe")

    # Handle different types of dataset_2
    if hasattr(dataset_2, 'column_names'):
        # It's a Dataset object
        sentences = dataset_2["Sentence"] if "Sentence" in dataset_2.column_names else []
        neutral_sentences = dataset_2["SentenceWithMask"] if "SentenceWithMask" in dataset_2.column_names else []
    else:
        # It's a dictionary-like object
        sentences = dataset_2.get("Sentence", [])
        neutral_sentences = dataset_2.get("SentenceWithMask", [])

    # Filter out None values and log info
    sentences = [s for s in sentences if s is not None]
    neutral_sentences = [s for s in neutral_sentences if s is not None]
    
    logger.info(f"Found {len(sentences)} sentences and {len(neutral_sentences)} neutral sentences")
    
    if len(sentences) == 0 and len(neutral_sentences) == 0:
        raise ValueError("No valid sentences found in dataset_2")

    # Get encodings for all sentences
    all_sentences = sentences + neutral_sentences
    all_encodings = lm.encode(all_sentences, result_type=result_type)

    if not only_generate_encodings:
        # Only use the non-neutral sentences for prediction and evaluation
        sentences_encodings = all_encodings[:len(sentences)]
        output_sentiments = probe.predict(sentences_encodings, sentences=sentences)
        logger.info(f"Generated predictions")

        return evaluate(dataset_2, output_sentiments)
