from dotenv import load_dotenv, find_dotenv


def main():
    import test_environment
    test_environment.check()

    import logging
    logger = logging.getLogger("src.main")

    logger.info("Starting main")

    # TODO obtain raw datasets to data/raw
    raise NotImplementedError("Can't obtain data")

    # TODO potentially process data - encodings of LM, minimal groups
    raise NotImplementedError("Can't process data")

    # TODO train probe on encodings of LM
    raise NotImplementedError("Can't train a probe on encodings of LM")

    # TODO evaluate model using minimal groups
    raise NotImplementedError("Can't evaluate model using minimal groups")


if __name__ == '__main__':
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    from src.utils.logging import setup_logger

    setup_logger()
    main()
