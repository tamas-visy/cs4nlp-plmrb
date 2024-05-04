import logging

from src.data.datatypes import SentimentData, GroupsDataset

logger = logging.getLogger(__name__)


def evaluate(truth: GroupsDataset, predicted: SentimentData):
    """Calculates metrics based on the real and predicted sentiments"""
    if len(truth) > 100_000:
        # TODO consider moving to polars if too many rows
        logger.warning("Consider moving to polars as pandas might be slow")

    df = truth.to_pandas()
    df["error"] = (df["label"] - predicted)
    df = df.groupby("group")
    mean = df['error'].mean()
    mean.name = "Mean error"
    mean = mean.sort_values()

    # TODO implement other metrics we promised

    return mean
