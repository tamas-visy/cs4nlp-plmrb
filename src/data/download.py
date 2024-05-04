import logging
import os.path
import shutil

import requests
import zipfile

from datasets import load_dataset

import pandas as pd
from tqdm import tqdm

from src.data.iohandler import IOHandler

logger = logging.getLogger(__name__)


class Downloader:
    @classmethod
    def all(cls, raise_notimplemented=True):
        cls.create_dummy()

        funcs = [cls.download_sst, cls.download_tweeteval,
                 cls.download_mdsd, cls.download_eec,
                 cls.download_honest, cls.download_labdet]
        for func in funcs:
            if raise_notimplemented:
                func()
            else:
                try:
                    func()
                except NotImplementedError:
                    logger.warning(f"{func.__name__} not implemented")

    @classmethod
    def download_sst(cls, force_download=False):
        """Downloads the Stanford Sentiment Treebank dataset."""
        if not os.path.exists(IOHandler.raw_path_to("sst2")) or force_download:
            load_dataset("sst2", cache_dir=IOHandler.raw_path_to(""))

    @classmethod
    def download_tweeteval(cls):
        """Downloads the TweetEval dataset."""
        raise NotImplementedError

    @classmethod
    def download_mdsd(cls):
        """Downloads the Multi-Domain Sentiment Dataset."""
        raise NotImplementedError

    @classmethod
    def download_eec(cls):
        """Downloads the Equity Evaluation Corpus."""
        raise NotImplementedError

    @classmethod
    def download_honest(cls):
        """Downloads the HONEST dataset."""
        raise NotImplementedError

    @classmethod
    def download_labdet(cls):
        """Downloads the data used in LABDet."""
        raise NotImplementedError

    @classmethod
    def create_dummy(cls):
        """Creates dummy data."""
        # Dataset
        data = [
            ("I love chocolate.", 0.8, 0.1, 0.0),
            ("I can't stand waiting in long lines.", 0.1, 0.8, 0.1),
            ("I hate when it rains.", 0.0, 0.9, 0.2),
            ("The weather in London is beautiful this time of the year", 0.8, 0.0, 0.1),
            ("That car is red.", 0.5, 0.5, 0.5),
            ("Wow, this is the best day ever!", 0.9, 0.0, 0.05),
            ("It's so sad to see the summer end.", 0.2, 0.1, 0.7),
            ("I love spending time with my family.", 0.85, 0.0, 0.1),
            ("Why do people have to be so mean sometimes?", 0.2, 0.8, 0.05)
        ]
        df = pd.DataFrame.from_records(data, columns=["Text", "Happy", "Angry", "Sad"])
        df.to_csv(IOHandler.raw_path_to("dummy.csv"))

        # Templates
        data = [
            ["{X} are really {Y}."],
            ["It is said that {X} can be {Y}."],
            ["Everyone agrees {X} are {Y}."],
            ["{X} are often {Y}."],
        ]
        df = pd.DataFrame.from_records(data, columns=["Template"])
        df.to_csv(IOHandler.raw_path_to("dummy_templates.csv"))
        # Groups
        data = dict(animals=["cats", "dogs", "Donkey and his magical friends",
                             "wolves", "sharks"],
                    people=["my family", "scientists", "US presidents",
                            "Shrek", "the absolutely worst people you can come up with"])
        df = pd.DataFrame.from_dict(data)
        # We expect the groups to have same length (for fairness when comparing), so we can save as columns of csv
        df.to_csv(IOHandler.raw_path_to("dummy_groups.csv"))
        # Adjectives
        data = [
            ["smelly", 0, 0, 1],
            ["funny", 1, 0, 0],
            ["cute", 1, 0, 0],
            ["neutral", 0, 1, 0],
        ]
        df = pd.DataFrame.from_records(data, columns=["Adjective", "Positive", "Neutral", "Negative"])
        df.to_csv(IOHandler.raw_path_to("dummy_adjectives.csv"))

    @classmethod
    def download_glove(cls, version):
        cls._download_and_extract_zip(f"GloVe{version.name}", url=version.value)

    @classmethod
    def _download_and_extract_zip(cls, name, url, force_download=False, force_extract=False):
        if not os.path.exists(IOHandler.raw_path_to(f"{name}.zip")) or force_download:
            logger.debug(f"Requesting {url} to {IOHandler.raw_path_to(f'{name}.zip')}")
            r = requests.get(url, stream=True)
            total = int(r.headers.get('content-length', 0))
            with open(IOHandler.raw_path_to(f"{name}.zip"), 'wb') as file:
                with tqdm(total=total, mininterval=1) as progress:
                    for data in r.iter_content(chunk_size=1024*1024):
                        written = file.write(data)
                        progress.update(written)
        if not os.path.exists(IOHandler.raw_path_to(name)) or force_extract:
            if os.path.exists(IOHandler.raw_path_to(name)):
                logger.debug(f"Deleting {IOHandler.raw_path_to(name)} as it already exists")
                shutil.rmtree(IOHandler.raw_path_to(name))
            logger.debug(f"Extracting zip to {IOHandler.raw_path_to(name)}")
            zipfile.ZipFile(IOHandler.raw_path_to(f"{name}.zip")).extractall(IOHandler.raw_path_to(name))

            logger.debug(f"Extracted zip to {IOHandler.raw_path_to(name)}")
