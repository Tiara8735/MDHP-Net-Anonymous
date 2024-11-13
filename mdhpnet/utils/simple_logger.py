import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from ruamel.yaml import YAML
import os
from os import path as osp

str2level = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def configLogger(
    name: str,
    level="INFO",
    to_console=True,
    save_path=None,
    mode="a",
    max_bytes=1048576,
    backup_count=3,
    log_fmt="[%(asctime)s][%(levelname)s][%(module)s:L%(lineno)d] >>> %(message)s"
) -> None:
    logger = logging.getLogger(name=name)
    logger.setLevel(level=str2level.get(level, logging.WARNING))
    formatter = logging.Formatter(log_fmt)
    if to_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(level=str2level.get(level, logging.WARNING))
        logger.addHandler(console)
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        rotating_file = ConcurrentRotatingFileHandler(
            filename=save_path,
            mode=mode,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        rotating_file.setFormatter(formatter)
        rotating_file.setLevel(level=str2level.get(level, logging.WARNING))
        logger.addHandler(rotating_file)


def initLoggers(cfgPath: str = "./configs/loggers.yml") -> None:
    """
    Initialize the logger.

    Parameters
    ----------
    cfgPath : str, optional
        The path to logger configuration file.
        Default: "<project-root-dir>/configs/logger.yml"
    """
    config = YAML().load(open(cfgPath, "r"))
    for k, v in config.items():
        configLogger(name=k, **v)
