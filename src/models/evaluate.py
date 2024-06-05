import logging

import pandas as pd

from src.data.datatypes import SentimentData, GroupedSubjectsDataset

logger = logging.getLogger(__name__)


def evaluate(truth: GroupedSubjectsDataset, predicted: SentimentData,
             show_subjects=None) -> pd.DataFrame:
    """Calculates metrics based on the real and predicted sentiments"""
    if len(truth) > 100_000:
        # TODO consider moving to polars if too many rows
        logger.warning("Consider moving to polars as pandas might be slow")

    df = truth.to_pandas()
    if show_subjects is None:
        # If we are working with less than 5 groups, show subjects grouped by groups, too
        show_subjects = len(df["group"].unique()) < 5

    # flip sign because we want a positive error if predicted is greater,
    #   but we need to subtract from the DataFrame / Series for type correctness
    df["error"] = -1 * (df["label"] - predicted)

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

    return mean.to_frame()  # convert pd.Series back to a DataFrame
