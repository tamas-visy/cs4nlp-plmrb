import logging

from src.data.datatypes import SentimentData, GroupedSubjectsDataset

logger = logging.getLogger(__name__)


def evaluate(truth: GroupedSubjectsDataset, predicted: SentimentData,
             show_subjects=True):
    """Calculates metrics based on the real and predicted sentiments"""
    if len(truth) > 100_000:
        # TODO consider moving to polars if too many rows
        logger.warning("Consider moving to polars as pandas might be slow")

    df = truth.to_pandas()
    df["error"] = -1 * (df["label"] - predicted)  # flip sign

    if show_subjects:
        df = df.groupby(["group", "subject"])
    else:
        df = df.groupby("group")
    mean = df['error'].mean()
    mean.name = "Mean error"
    mean = mean.sort_values()
    if show_subjects:
        mean = mean.sort_index(level=0, sort_remaining=False)

    # TODO implement other metrics we promised

    return mean
