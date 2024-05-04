import logging
import os


def setup_logger(level=None):
    if level is None:
        level = logging.getLevelNamesMapping()[os.getenv("LOGGING_LEVEL", default=logging.getLevelName(logging.INFO))]
    logging_format = '{asctime}.{msecs:03.0f} | {name:^64s} | {levelname:.3s} | {message}'
    date_format = '%d-%m-%Y %H:%M:%S'

    logging.basicConfig(level=logging.WARNING,
                        format=logging_format,
                        style='{',
                        datefmt=date_format)

    logger = logging.getLogger(__name__)
    logger.debug("Logger created")

    # We specifically set our logger to level, the rest to INFO
    logging.getLogger("src").setLevel(level)


