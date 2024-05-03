import logging

import pandas as pd

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
    def download_sst(cls):
        """Downloads the Stanford Sentiment Treebank dataset."""
        raise NotImplementedError

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
            ("I hate when it rains.", 0.0, 0.9, 0.2),
            ("That car is red.", 0.5, 0.5, 0.5),
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
        # Subjects
        data = [
            ["cats"],
            ["dogs"],
            ["my family"],
        ]
        df = pd.DataFrame.from_records(data, columns=["Subject"])
        df.to_csv(IOHandler.raw_path_to("dummy_subjects.csv"))
        # Adjectives
        data = [
            ["smelly", 0, 0, 1],
            ["funny", 1, 0, 0],
            ["cute", 1, 0, 0],
            ["neutral", 0, 1, 0],
        ]
        df = pd.DataFrame.from_records(data, columns=["Adjective", "Positive", "Neutral", "Negative"])
        df.to_csv(IOHandler.raw_path_to("dummy_adjectives.csv"))
