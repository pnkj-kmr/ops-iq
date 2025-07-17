import logging
from loguru import logger
import sys
from config.settings import settings

# Configure loguru
logger.remove()
logger.add(sys.stderr, level=settings.log_level, format="{time} | {level} | {message}")
logger.add(settings.log_file, rotation="100 MB", retention="10 days", level=settings.log_level)

def get_logger(name: str):
    """Get logger instance"""
    return logger.bind(module=name)
