import os
import logging

def create_logger(logger_name, log_path = None):
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if log_path is None:
        handler = logging.StreamHandler() # show log in console
    else:
        handler = logging.FileHandler(log_path) # print log in file
    
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            fmt = '%(asctime)s %(levelname)s:  %(message)s',
            datefmt ='%m-%d %H:%M'
        )
    )
    logger.addHandler(handler)

    return logger

