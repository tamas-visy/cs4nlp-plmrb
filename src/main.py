from dotenv import load_dotenv, find_dotenv


def main():
    import environment
    if not environment.get_flag("SKIP_VERIFYING_ENVIRONMENT"):
        environment.verify()
    DEVELOP_MODE = environment.get_flag("DEVELOP_MODE")

    import logging
    logger = logging.getLogger("src.main")

    logger.info("Starting main")
    if DEVELOP_MODE:
        logger.warning("DEVELOP_MODE is enabled")

    # Obtain raw datasets to data/raw
    from src.data.download import Downloader
    Downloader.all(raise_notimplemented=False)
    logger.debug("Downloaded data")

    from src.data.iohandler import IOHandler
    from datasets import concatenate_datasets
    # dataset_1 = IOHandler.load_dummy_dataset()
    # Note: to concatenate datasets, they must have compatible features
    dataset_1 = concatenate_datasets(
        [IOHandler.load_sst(),
         IOHandler.load_tweeteval()])
    logger.info(f"Loaded dataset with {len(dataset_1)} rows")

    from src.data.clean import clean_dataset
    dataset_1 = clean_dataset(dataset_1, dummy=True)  # TODO use proper version
    logger.info(f"Cleaned dataset, {len(dataset_1)} rows remaining")
    # TODO save cleaned dataset

    if DEVELOP_MODE:
        dataset_1 = dataset_1.shuffle(seed=42).select(range(1000))
        logger.debug(f"Subsampled data to {len(dataset_1)} rows")

    # Process data
    from src.models.language_model import LanguageModel, BERTLanguageModel
    lm: LanguageModel = BERTLanguageModel()
    from src.data.datatypes import EncodingData
    encodings: EncodingData = lm.encode(dataset_1["input"])  # TODO potentially save encodings

    # Train probe on encodings of LM
    from src.models.probe import Probe, MLPProbe
    from datasets import Dataset
    probe: Probe = MLPProbe()
    probe.train(dataset=Dataset.from_dict(dict(input=encodings, label=dataset_1["label"])))
    # TODO potentially save trained probe
    logger.info(f"Trained probe")

    # Evaluate encodings of LM using the probe
    dataset_2: Dataset
    if DEVELOP_MODE:
        # We have a good expectation of how these subjects should be ordered
        #   so we evaluate them when in DEVELOP_MODE
        from src.data.generate import generate
        templates = IOHandler.load_dummy_templates()
        groups = IOHandler.load_dummy_groups()
        adjectives = IOHandler.load_dummy_adjectives()
        dummy_generated = generate(templates, groups, adjectives)
        dataset_2 = dummy_generated
    else:
        dataset_2 = IOHandler.load_labdet_test()

    if DEVELOP_MODE:
        dataset_2 = dataset_2.shuffle(seed=42).select(range(100))
        logger.debug(f"Subsampled data to {len(dataset_2)} rows")

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
