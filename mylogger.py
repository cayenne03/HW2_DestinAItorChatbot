import logging

def get_logger(name=None):
    # create logger
    logger = logging.getLogger(name or 'DestinAItorChatbot')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # 🚫 Prevent logs from propagating to the root logger

    # prevent adding handlers multiple times
    if not logger.handlers:
        # create handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        # create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        # add handler to logger
        logger.addHandler(handler)

    return logger

# create and export logger instance
logger = get_logger()