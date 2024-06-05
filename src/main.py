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
        logger.debug(f"Subsampled dataset #1 to {len(dataset_1)} rows")

    # Evaluate encodings of LM using the probe
    from datasets import Dataset
    dataset_2: Dataset
    if DEVELOP_MODE:
        # We have a good expectation of how these subjects should be ordered,
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
        logger.debug(f"Subsampled dataset #2 to {len(dataset_2)} rows")
    logger.info(f"Generated {len(dataset_2)} sentences")

    from src.models.language_model import TransformerModel, GPT2LanguageModel
    from src.models.probe import Probe, MLPProbe
    from typing import List, Type, Dict, Any, Literal
    from pandas import DataFrame
    import src.process as process
    results: Dict[Any, DataFrame] = dict()

    # TODO also evaluate some base

    lm_factories: List[Type[TransformerModel]] = [GPT2LanguageModel]
    result_types: List[int | Literal["initial", "final", "middle"]] = ["initial", "final"]
    probe_factory: Type[Probe] = MLPProbe
    for lm_factory in lm_factories:
        for result_type in result_types:
            results[(lm_factory, result_type)] = process.complete(
                    lm_factory=lm_factory,
                    result_type=result_type,
                    probe_factory=probe_factory,
                    dataset_1=dataset_1,
                    dataset_2=dataset_2
                )

    # Evaluate model encodings
    logger.debug(f"Evaluated sentiments")
    logger.info(f"The results are:\n{results}")


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    from src.utils.logging import setup_logger

    setup_logger()
    main()
