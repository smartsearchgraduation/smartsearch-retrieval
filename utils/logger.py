import logging
from config import Config


def get_logger(name="retrieval_system"):
    logger = logging.getLogger(name)
    logger.setLevel(Config.LOG_LEVEL)
    ch = logging.StreamHandler()
    ch.setLevel(Config.LOG_LEVEL)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(ch)
    return logger


logger = get_logger()
