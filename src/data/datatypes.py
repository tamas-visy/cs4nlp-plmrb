from typing import List

import numpy as np
from datasets import Dataset


TextData = List[str]
"""The abstracted away type we will use as input for generating encodings, consisting of strings"""

SentimentData = List[np.ndarray]
"""The abstracted away type we will use as labels for generated encodings, consisting of sentiments"""

EncodingData = List[np.ndarray]
"""The abstracted away type we will use as input for the probe, consisting of encodings"""

# TODO handle when some data is removed, so maybe use a pd.DataFrame so we can use it's index to connect X and Y

TextDataset = Dataset
"""The abstracted away type we get, consisting of encodings and sentiments. Features are input and label"""


ProbeDataset = Dataset
"""The abstracted away type we will use as a dataset for training the probe, consisting of encodings and sentiments.
Features are input and label"""

GroupsDataset = Dataset
"""The abstracted away type we will use as a dataset for evaluating the probe, consisting of groups of subjects
 and sentiments. Features are group and label"""
