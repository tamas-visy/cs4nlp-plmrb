import json
import logging
import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, ClassLabel

from src.data.datatypes import TextDataset, EncodingDataset, GroupedSubjectsDataset

logger = logging.getLogger(__name__)


class IOHandler:
    _path_raw = "data/raw"
    _path_interim = "data/interim"
    _path_processed = "data/processed"

    @classmethod
    def raw_path_to(cls, target):
        return f"{cls._path_raw}/{target}"

    @classmethod
    def interim_path_to(cls, target):
        return f"{cls._path_interim}/{target}"

    @classmethod
    def processed_path_to(cls, target):
        return f"{cls._path_processed}/{target}"

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
        dataset = dataset.remove_columns(["idx"])
        dataset = dataset.rename_column("sentence", "input")
        dataset = dataset.rename_columns(dict())["train"]
        return dataset

    @classmethod
    def load_tweeteval(cls) -> TextDataset:
        # Negative would be 0
        TWEETEVAL_NEUTRAL_LABEL, TWEETEVAL_POSITIVE_LABEL = 1, 2
        dataset = load_dataset(IOHandler.raw_path_to("tweeteval"))

        def _convert_positive_to_one(row):
            if row["label"] == TWEETEVAL_POSITIVE_LABEL:
                row["label"] = 1  # "positive"
            return row

        # Remove neutral rows and update the features
        dataset = dataset.filter(lambda row: row["label"] != TWEETEVAL_NEUTRAL_LABEL)
        dataset = dataset.cast_column("label", ClassLabel(num_classes=2, names=['negative', 'positive']))

        dataset = dataset.map(_convert_positive_to_one)
        dataset = dataset.rename_columns(dict(text="input"))["test"]
        return dataset

    @classmethod
    def load_labdet_test(cls) -> EncodingDataset | GroupedSubjectsDataset:
        """Loads the english LABDet test set, which contains different nationalities with neutral adjectives."""
        dataset = Dataset.from_json(IOHandler.raw_path_to("LABDet/LABDet-main/test/en.json"))
        with open(IOHandler.raw_path_to("LABDet/LABDet-main/Templates/en_template.json")) as f:
            data = json.load(f)
        adjective_map = dict()
        pos_adj = data["sentiment_templates"][0]["pos_adj"]
        for adj in pos_adj:
            adjective_map[adj] = 1
        neg_adj = data["sentiment_templates"][0]["neg_adj"]
        for adj in neg_adj:
            adjective_map[adj] = 0
        neutral_adj = data["artificial_experiments"]["neutral_adj"]
        for adj in neutral_adj:
            adjective_map[adj] = 0.5

        nationality_map = data["alternatives"]

        dataset = dataset.add_column(name="label",
                                     column=[adjective_map[dataset[i]["adj"]] for i in range(len(dataset))])
        dataset = dataset.add_column(name="group",
                                     column=dataset["nationality"])

        def nationality_map_func(row):
            row.update({"nationality": nationality_map[row["nationality"]]})
            return row

        dataset = dataset.map(nationality_map_func)
        dataset = dataset.rename_columns(dict(sentence="input", nationality="subject"))
        # dataset = dataset.filter(lambda row: "mask" not in row["group"])
        return dataset

    @classmethod
    def get_dataset_1(cls, develop_mode=False) -> TextDataset:
        """Loads dataset 1, using cached files if available."""
        processed_dataset_1_path = IOHandler.processed_path_to("train_dataset_processed.csv")
        if os.path.exists(processed_dataset_1_path):
            dataset_1 = Dataset.from_csv(processed_dataset_1_path)
            logger.info(f"Found processed dataset with {len(dataset_1)} rows")
        else:
            interim_dataset_1_path = IOHandler.interim_path_to("train_dataset_interim.csv")
            if os.path.exists(interim_dataset_1_path):
                dataset_1 = Dataset.from_csv(interim_dataset_1_path)
                logger.info(f"Found interim dataset with {len(dataset_1)} rows")
            else:
                from datasets import concatenate_datasets
                # dataset_1 = IOHandler.load_dummy_dataset()
                # Note: to concatenate datasets, they must have compatible features
                dataset_1 = concatenate_datasets(
                    [IOHandler.load_sst(),
                     IOHandler.load_tweeteval()])
                logger.info(f"Loaded dataset with {len(dataset_1)} rows")

                if develop_mode:
                    dataset_1 = dataset_1.shuffle(seed=42).select(range(1000))
                    logger.debug(f"Subsampled dataset #1 to {len(dataset_1)} rows")
                pd.DataFrame(dataset_1).to_csv(interim_dataset_1_path, index=False)
                logger.info(f"Saved interim dataset to {interim_dataset_1_path}")

            from src.data.clean import clean_dataset
            dataset_1 = clean_dataset(dataset_1)
            logger.info(f"Cleaned dataset, {len(dataset_1)} rows remaining")
            pd.DataFrame(dataset_1).to_csv(processed_dataset_1_path, index=False)
            logger.info(f"Saved processed dataset to {processed_dataset_1_path}")
        return dataset_1
