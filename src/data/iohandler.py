import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset

from src.data.datatypes import TextDataset

logger = logging.getLogger(__name__)


class IOHandler:
    _path_raw = "data/raw"
    _path_interim = "data/interim"

    @classmethod
    def raw_path_to(cls, target):
        return f"{cls._path_raw}/{target}"

    @classmethod
    def load_dummy_dataset(cls, raw=True) -> TextDataset:
        if raw:
            df = pd.read_csv(cls.raw_path_to("dummy.csv"), index_col=0)
            df['label'] = df[['Happy', 'Angry', 'Sad']].apply(np.array, axis=1)
            df = df.drop(['Happy', 'Angry', 'Sad'], axis="columns")
            df.index.name = "index"
            return Dataset.from_pandas(df).rename_column("Text", "input")
        else:
            # TODO how to best handle caching of intermediate results
            raise NotImplementedError

    @classmethod
    def load_dummy_templates(cls) -> List[str]:
        df = pd.read_csv(cls.raw_path_to("dummy_templates.csv"), index_col=0)
        return df["Template"].values.tolist()

    @classmethod
    def load_dummy_groups(cls) -> Dict[str, List[str]]:
        df = pd.read_csv(cls.raw_path_to("dummy_groups.csv"), index_col=0)
        return df.to_dict(orient="list")

    @classmethod
    def load_dummy_adjectives(cls) -> Tuple[List[str], List[str], List[str]]:
        df = pd.read_csv(cls.raw_path_to("dummy_adjectives.csv"), index_col=0)
        pos = df["Adjective"][df["Positive"] == 1]
        neut = df["Adjective"][df["Neutral"] == 1]
        neg = df["Adjective"][df["Negative"] == 1]
        return pos.values.tolist(), neut.values.tolist(), neg.values.tolist()

    @classmethod
    def load_glove_embeddings(cls, version, embedding_dim) -> Dict[str, np.ndarray]:
        logger.debug(f"Loading GloVe embeddings "
                     f"from {cls.raw_path_to(f'GloVe{version.name}/glove.6B.{embedding_dim}d.txt')}")
        embeddings_dict = {}
        with open(cls.raw_path_to(f"GloVe{version.name}/glove.6B.{embedding_dim}d.txt"), encoding="utf-8") as f:
            for line in f:
                values = line.split()
                embeddings_dict[values[0]] = np.asarray(values[1:], dtype=float)
        logger.debug(f"Loaded GloVe embeddings")
        return embeddings_dict

    @classmethod
    def load_sst(cls) -> TextDataset:
        dataset = load_dataset(IOHandler.raw_path_to("sst2"))
        return dataset.rename_columns(dict(sentence="input", idx="index"))["train"]