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
        # raise RuntimeError("Running with DEVELOP_MODE is not supported anymore")  # TODO comment in

    # Obtain raw datasets to data/raw
    from src.data.download import Downloader

    Downloader.all(raise_notimplemented=False)
    logger.debug("Downloaded data")

    from datasets import Dataset
    from src.data.iohandler import IOHandler

    dataset_1: Dataset = IOHandler.get_dataset_1(develop_mode=DEVELOP_MODE)

    # Evaluate encodings of LM using the probe
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
        logger.info(f"Generated {len(dataset_2)} sentences")
    else:
        dataset_2 = IOHandler.get_dataset_2()

    if DEVELOP_MODE:
        dataset_2 = dataset_2.shuffle(seed=42).select(range(100))
        logger.debug(f"Subsampled dataset #2 to {len(dataset_2)} rows")

    from typing import List, Type, Literal
    import pandas as pd
    from src.models.language_model import (
        TransformerModel,
        GPT2LanguageModel,
        GloveLanguageModel,
        BERTLanguageModel,
        LLaMALanguageModel,
        RoBERTaLanguageModel,
        ELECTRALanguageModel,
        T5LanguageModel,
        XLNetLanguageModel,
    )
    from src.models.probe import Probe, MLPProbe
    import src.process as process

    results: List[pd.DataFrame] = []

    # TODO also evaluate some baseline

    lms: List[TransformerModel] = [
        # GPT2LanguageModel(),  # idx: G
        BERTLanguageModel(),  # idx: B
        # LLaMALanguageModel(), #idx: L
        # RoBERTaLanguageModel(),  # idx: R
        # ELECTRALanguageModel(),  # idx: E
        # GloveLanguageModel(),  # idx: G
        # T5LanguageModel(),  # idx: T
        # XLNetLanguageModel(),  # idx: X
    ]
    probe_factory: Type[Probe] = MLPProbe
    for lm in lms:
        result_types: List[int | Literal["initial", "final", "middle"]] = [
            "initial",
            "middle",  # "*list(range(lm.num_encoder_layers + 1)),
            "final",
        ]
        for result_type in result_types:
            result = process.complete(
                lm=lm,
                result_type=result_type,
                probe_factory=probe_factory,
                dataset_1=dataset_1,
                dataset_2=dataset_2,
                only_generate_encodings=True,  # enable this to skip probe training
            )
            if result is not None:
                result["value"] = lm.__class__.__name__
                result["result_type"] = result_type
                try:
                    result = result.reset_index().set_index(
                        ["value", "result_type", "group", "subject"]
                    )
                except KeyError:
                    # happens if subject not in the columns
                    result = result.reset_index()
                    result["subject"] = result[
                        "group"
                    ]  # this is not really valid, but it's a workaround
                    result = result.set_index(
                        ["value", "result_type", "group", "subject"]
                    )
                results.append(result)

    results: pd.DataFrame = pd.concat(results)

    # Evaluate model encodings
    logger.debug(f"Evaluated sentiments")
    logger.info(f"The results are:\n{results}")

    with open("out/results.json", "w") as f:
        results.to_json(f, indent=4)


if __name__ == "__main__":
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    import sys
    import os

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.utils.logging import setup_logger

    setup_logger()
    main()
