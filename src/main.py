from dotenv import load_dotenv, find_dotenv


def main():
    import test_environment
    test_environment.check()

    import logging
    logger = logging.getLogger("src.main")

    logger.info("Starting main")

    # Obtain raw datasets to data/raw
    from src.data.download import Downloader
    Downloader.all()

    from src.data.datasets import TextData, SentimentData, EncodingData
    data: TextData = None  # TODO load
    sentiments: SentimentData = None  # TODO load

    # Process data
    from src.models.language_model import LanguageModel
    lm: LanguageModel = None  # TODO create
    encodings: EncodingData = lm.encode(data)  # TODO potentially save encodings

    # Train probe on encodings of LM
    from src.models.probe import Probe
    probe: Probe = None  # TODO create
    probe.train(dataset=(sentiments, encodings))  # TODO potentially save trained probe

    # Evaluate encodings of LM using the probe
    from src.data.generate import generate
    texts, sentiments = generate(templates=None, subjects=None, adjectives=None)
    encodings = lm.encode(texts)
    output_sentiments = probe.predict(encodings)  # TODO potentially save output sentiments

    # Evaluate model encodings
    from src.models.evaluate import evaluate
    results = evaluate(sentiments, output_sentiments)
    logger.info(str(results))


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    from src.utils.logging import setup_logger

    setup_logger()
    main()
