import logging
from typing import Literal, Type

import pandas as pd
from datasets import Dataset

from src.data.datatypes import TextDataset, EncodingData, EncodingDataset
from src.models.evaluate import evaluate
from src.models.language_model import TransformerModel
from src.models.probe import Probe

logger = logging.getLogger(__name__)


def complete(lm_factory: Type[TransformerModel],
             result_type: int | Literal["initial", "final", "middle"],
             probe_factory: Type[Probe],
             dataset_1: TextDataset,
             dataset_2: EncodingDataset) -> pd.DataFrame:
    # Setup instances used for testing
    lm = lm_factory()  # noqa  # child classes implement
    probe = probe_factory()  # noqa  # child classes implement

    encodings: EncodingData = lm.encode(dataset_1["input"], result_type=result_type)  # TODO potentially save encodings

    probe.train(dataset=Dataset.from_dict(dict(input=encodings, label=dataset_1["label"])))
    # TODO potentially save trained probe
    logger.info(f"Trained probe")

    encodings = lm.encode(dataset_2["input"])  # TODO save LM encodings of templates
    output_sentiments = probe.predict(encodings)  # TODO potentially save output sentiments
    logger.info(f"Generated predictions")

    return evaluate(dataset_2, output_sentiments)
