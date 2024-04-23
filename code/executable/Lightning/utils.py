import logging
import numpy as np
from lightning.pytorch.callbacks import TQDMProgressBar

# Class definition


class CustomProgressBar(TQDMProgressBar):
    """
    Customized lightning progress bar
    """

    def init_sanity_tqdm(self):
        bar = super().init_sanity_tqdm()
        bar.colour = "#00ff00"
        return bar

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.colour = "#f50000"
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.colour = "#ffaa01"
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        bar.colour = "#fcff13"
        return bar


class CustomLogFormatter(logging.Formatter):
    """
    Customized logging formatter
    """

    green = "\x1b[32m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: green + fmt + reset,
        logging.INFO: green + fmt + reset,
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Function definition


def getLogger(level=logging.INFO) -> logging.Logger:
    """Initializing and getting a logger

    Args:
        level (int): level to use (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR or logging.CRITICAL)

    Returns:
        logging.RootLogger: an initialized customized logger

    """
    # Creating logger
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    # Creating console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setFormatter(CustomLogFormatter())
    logger.addHandler(ch)
    return logger
