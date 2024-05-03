from typing import List, Tuple

import pandas as pd

from src.data.datasets import TextDataset


class IOHandler:
    path_raw = "data/raw"
    path_interim = "data/interim"

    @classmethod
    def raw_path_to(cls, target):
        return f"{cls.path_raw}/{target}"

    @classmethod
    def load_dummy_dataset(cls, raw=True) -> TextDataset:
        if raw:
            df = pd.read_csv(cls.raw_path_to("dummy.csv"), index_col=0)
            texts = df["Text"].to_list()
            sentiments = df[["Happy", "Angry", "Sad"]].values.tolist()  # TODO maybe don't convert from numpy to list
            return texts, sentiments
        else:
            # TODO how to best handle caching of intermediate results
            raise NotImplementedError

    @classmethod
    def load_dummy_templates(cls) -> List[str]:
        df = pd.read_csv(cls.raw_path_to("dummy_templates.csv"), index_col=0)
        return df["Template"].values.tolist()

    @classmethod
    def load_dummy_subjects(cls) -> List[str]:
        df = pd.read_csv(cls.raw_path_to("dummy_subjects.csv"), index_col=0)
        return df["Subject"].values.tolist()

    @classmethod
    def load_dummy_adjectives(cls) -> Tuple[List[str], List[str], List[str]]:
        df = pd.read_csv(cls.raw_path_to("dummy_adjectives.csv"), index_col=0)
        pos = df["Adjective"][df["Positive"] == 1]
        neut = df["Adjective"][df["Neutral"] == 1]
        neg = df["Adjective"][df["Negative"] == 1]
        return pos.values.tolist(), neut.values.tolist(), neg.values.tolist()
