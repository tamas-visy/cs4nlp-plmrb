from typing import List, Tuple, Dict

from dotenv import load_dotenv, find_dotenv


def main():
    import environment
    if not environment.get_flag("SKIP_VERIFYING_ENVIRONMENT"):
        environment.verify()

    import logging
    logger = logging.getLogger("src.main")

    logger.info("Starting main")

    # Obtain raw datasets to data/raw
    from src.data.download import Downloader
    Downloader.all(raise_notimplemented=False)
    logger.debug("Downloaded data")

    from src.data.iohandler import IOHandler
    dataset_1 = IOHandler.load_dummy_dataset()
    dataset_1 = IOHandler.load_sst()
    logger.info(f"Loaded dataset with {len(dataset_1)} rows")

    from src.data.clean import clean_dataset
    dataset_1 = clean_dataset(dataset_1, dummy=True)  # TODO use proper version
    logger.info(f"Cleaned dataset, {len(dataset_1)} rows remaining")
    # TODO save cleaned dataset

    # Process data
    from src.models.language_model import LanguageModel, GloveLanguageModel
    lm: LanguageModel = GloveLanguageModel()
    from src.data.datatypes import EncodingData
    encodings: EncodingData = lm.encode(dataset_1["input"])  # TODO potentially save encodings

    # Train probe on encodings of LM
    from src.models.probe import Probe, LinearProbe
    from datasets import Dataset
    probe: Probe = LinearProbe()  # TODO use proper Probe
    probe.train(dataset=Dataset.from_dict(dict(input=encodings, label=dataset_1["label"])))
    # TODO potentially save trained probe
    logger.info(f"Trained probe")

    # Evaluate encodings of LM using the probe
    from src.data.generate import generate
    # TODO use not dummy values
    templates: List[str] = IOHandler.load_dummy_templates()
    groups: Dict[str, List[str]] = IOHandler.load_dummy_groups()
    adjectives: Tuple[List[str], List[str], List[str]] = IOHandler.load_dummy_adjectives()

    dataset_2 = generate(templates, groups, adjectives)
    logger.info(f"Generated {len(dataset_2)} sentences")
    encodings = lm.encode(dataset_2["input"])  # TODO save LM encodings of templates
    output_sentiments = probe.predict(encodings)  # TODO potentially save output sentiments
    logger.info(f"Generated predictions")

    # Evaluate model encodings
    from src.models.evaluate import evaluate
    results = evaluate(dataset_2, output_sentiments)
    logger.debug(f"Evaluated sentiments")
    logger.info(f"The results are:\n{results}")


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    from src.utils.logging import setup_logger

    setup_logger()
    main()
