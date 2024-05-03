from typing import List, Tuple

from dotenv import load_dotenv, find_dotenv


def main():
    import test_environment
    test_environment.check()

    import logging
    logger = logging.getLogger("src.main")

    logger.info("Starting main")

    # Obtain raw datasets to data/raw
    from src.data.download import Downloader
    Downloader.all(raise_notimplemented=False)  # TODO use proper one
    logger.debug("Downloaded data")

    from src.data.datasets import TextData, SentimentData, EncodingData
    from src.data.iohandler import IOHandler
    dataset = IOHandler.load_dummy_dataset()  # TODO load proper dataset
    logger.debug(f"Loaded dataset with {len(dataset[0])} inputs and {len(dataset[1])} labels")

    from src.data.clean import clean_dataset
    dataset = clean_dataset(dataset, dummy=True)  # TODO use proper version
    logger.debug(f"Cleaned dataset, remaining inputs: {len(dataset[0])}")
    # TODO save cleaned dataset

    texts: TextData = dataset[0]
    sentiments: SentimentData = dataset[1]

    # Process data
    from src.models.language_model import LanguageModel, DummyLanguageModel
    lm: LanguageModel = DummyLanguageModel()  # TODO use proper LM
    encodings: EncodingData = lm.encode(texts)  # TODO potentially save encodings
    logger.debug(f"Generated {len(encodings)} encodings")

    # Train probe on encodings of LM
    from src.models.probe import Probe, DummyProbe
    probe: Probe = DummyProbe()  # TODO use proper Probe
    probe.train(dataset=(encodings, sentiments))  # TODO potentially save trained probe
    logger.debug(f"Trained probe")

    # Evaluate encodings of LM using the probe
    from src.data.generate import generate
    # TODO use not dummy values
    templates: List[str] = IOHandler.load_dummy_templates()
    subjects: List[str] = IOHandler.load_dummy_subjects()
    adjectives: Tuple[List[str], List[str], List[str]] = IOHandler.load_dummy_adjectives()

    dataset = generate(templates, subjects, adjectives)
    logger.debug(f"Generated {len(dataset[0])} sentences")
    texts, sentiments = dataset
    encodings = lm.encode(texts)  # TODO save LM encodings of templates
    output_sentiments = probe.predict(encodings)  # TODO potentially save output sentiments
    logger.debug(f"Generated predictions")

    # Evaluate model encodings
    from src.models.evaluate import evaluate
    results = evaluate(sentiments, output_sentiments, dummy=True)  # TODO use proper evaluation
    logger.debug(f"Evaluated sentiments")
    logger.info(f"The results are {str(results)}")


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    from src.utils.logging import setup_logger

    setup_logger()
    main()
