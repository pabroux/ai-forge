import logging

import torch

# Class definition


class CustomLogFormatter(logging.Formatter):
    """
    Custom Logging Formatter
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


def get_logger() -> logging.Logger:
    """Initializing and getting a logger

    Args:
        None

    Returns:
        logging.RootLogger: an initialized customized logger

    """
    ## Creating logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    ## Creating console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setFormatter(CustomLogFormatter())
    logger.addHandler(ch)
    return logger


def chooseDevice(device: str, logger: logging.Logger = None) -> torch.device:
    """Choosing the device to use with the network

    Args:
        device (str): the device we would like to use ("mps" or "cpu")
        logger (logging.Logger): a specific logger to use

    Returns:
        torch.device: the chosen torch.device object

    """
    if logger is None:
        logger = get_logger()
    if device == "mps" and torch.backends.mps.is_available():
        logger.info("MPS as device")
        torch_device = torch.device("mps")
    else:
        if device != "cpu":
            logger.warning("CPU as device. GPU not available")
        else:
            logger.info("CPU as device")
        torch_device = torch.device("cpu")
    return torch_device
