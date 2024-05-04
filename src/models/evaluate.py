import logging

from src.data.datatypes import SentimentData, SubjectDataset

logger = logging.getLogger(__name__)


def evaluate(truth: SubjectDataset, predicted: SentimentData):
    """Calculates metrics based on the real and predicted sentiments"""
    if len(truth) > 100_000:
        # TODO consider moving to polars if too many rows
        logger.warning("Consider moving to polars as pandas might be slow")

    df = truth.to_pandas()
    df["error"] = (df["label"] - predicted)**2  # squared errors
    df = df.groupby("subject")
    rmse = df['error'].mean()**0.5  # root mean [error]
    rmse.name = "RMSE"
    rmse = rmse.sort_values()

    # TODO implement other metrics we promised

    return rmse
