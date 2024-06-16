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
    # Setup instances used for testing
    probe = probe_factory()  # noqa  # child classes implement it

    encodings: EncodingData = lm.encode(dataset_1["input"], result_type=result_type)

    if not only_generate_encodings:
        probe.train(
            dataset=Dataset.from_dict(dict(input=encodings, label=dataset_1["label"]))
        )
        # TODO potentially save trained probe
        logger.info(f"Trained probe")

    sentences = dataset_2["input"]
    sentences.extend(dataset_2["input_neutral"])

    encodings = lm.encode(sentences, result_type=result_type)

    if not only_generate_encodings:
        output_sentiments = probe.predict(encodings, sentences=sentences)
        logger.info(f"Generated predictions")

        return evaluate(dataset_2, output_sentiments)
