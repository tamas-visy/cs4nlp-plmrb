from typing import List, Tuple

import numpy as np

TextData = List[str]
"""The abstracted away type we will use as input for generating encodings, consisting of strings"""

SentimentData = List[np.ndarray]
"""The abstracted away type we will use as labels for generated encodings, consisting of sentiments"""

EncodingData = List[np.ndarray]
"""The abstracted away type we will use as input for the probe, consisting of encodings"""

# TODO handle when some data is removed, so maybe use a pd.DataFrame so we can use it's index to connect X and Y

ProbeDataset = Tuple[EncodingData, SentimentData]
"""The abstracted away type we will use as a dataset for training the probe, consisting of encodings and sentiments"""
