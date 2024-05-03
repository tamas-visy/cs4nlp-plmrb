def download_sst():
    """Downloads the Stanford Sentiment Treebank dataset."""
    raise NotImplementedError


def download_tweeteval():
    """Downloads the TweetEval dataset."""
    raise NotImplementedError


def download_mdsd():
    """Downloads the Multi-Domain Sentiment Dataset."""
    raise NotImplementedError


def download_eec():
    """Downloads the Equity Evaluation Corpus."""
    raise NotImplementedError


def download_honest():
    """Downloads the HONEST dataset."""
    raise NotImplementedError


def download_labdet():
    """Downloads the data used in LABDet."""
    raise NotImplementedError


class Downloader:
    @classmethod
    def all(cls):
        download_sst()
        download_tweeteval()
        download_mdsd()
        download_eec()
        download_honest()
        download_labdet()
