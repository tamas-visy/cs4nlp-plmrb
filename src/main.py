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

    if DEVELOP_MODE:
        dataset_1 = dataset_1.shuffle(seed=42).select(range(1000))
        logger.debug(f"Subsampled dataset #1 to {len(dataset_1)} rows")

    from src.data.clean import clean_dataset
    dataset_1 = clean_dataset(dataset_1)  # TODO use proper version
    logger.info(f"Cleaned dataset, {len(dataset_1)} rows remaining")
    # TODO save cleaned dataset

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

    from typing import List, Type, Literal
    import pandas as pd
    from src.models.language_model import TransformerModel, GPT2LanguageModel, BERTLanguageModel
    from src.models.probe import Probe, MLPProbe
    import src.process as process
    results: List[pd.DataFrame] = []

    # TODO also evaluate some baseline

    lm_factories: List[Type[TransformerModel]] = [GPT2LanguageModel]
    result_types: List[int | Literal["initial", "final", "middle"]] = ["initial", "middle", "final"]
    probe_factory: Type[Probe] = MLPProbe
    for lm_factory in lm_factories:
        for result_type in result_types:
            result = process.complete(
                lm_factory=lm_factory,
                result_type=result_type,
                probe_factory=probe_factory,
                dataset_1=dataset_1,
                dataset_2=dataset_2
            )
            result['lm'] = lm_factory.__name__
            result['result_type'] = result_type
            result = result.reset_index().set_index(['lm', 'result_type', 'group', 'subject'])
            results.append(result)

    results: pd.DataFrame = pd.concat(results)

    # Evaluate model encodings
    logger.debug(f"Evaluated sentiments")
    logger.info(f"The results are:\n{results}")

    print(results.to_json(indent=4))


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    from src.utils.logging import setup_logger

    setup_logger()
    main()
