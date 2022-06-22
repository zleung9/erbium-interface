import os
import numpy as np
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


def vertical_sum(value_list, mean=True):
    """
    Given a list of signs arrays zeroed at the first non-zero sign. e.g.:
    [
        [1,1,0,1,0,1,0,0,1,0,1,1,1,1,1],
        [1,0,1,0,0,0,0,0],
        [1,1,0,1,1,0,0,1,0,0,1,1,0,1,],
        ...
    ]
    where each list have different length, sum them in the vertical direction and get:
        [3, 2, 1, 2, 1, 1, 0, 1, 1, 0, 2, 2, 1, 2, 1]
    or average them and get:
        [1, 2/3, 1/3, 2/3, 1/3, 1/3, 0, 1/3, 1/2, 0, 1, 1, 1/2, 1, 1]

    Parameters
    ----------
    value_list : list
        The input list of lists (see the example above). The sublist can contain any number.
    mean : bool
        If true return mean, otherwise return sum.

    Returns
    -------
    max_length : int
        The maximum length of signs (15 in the above example)
    value_sum : list
        A list whose element is the sum/mean in each column.
    value_std : list
        A list whose element is the standard deviation in each column.
    """

    # maximum length of signs array used as the tota span 
    max_length = max([len(v) for v in value_list])
    
    value_sum = []
    value_std = []
    
    for i in range(max_length): # frame from the first nonzero index
        value_list_vertical = []
        for v in value_list: # get all the ith element of value list, if any
            try:
                value_list_vertical.append(v[i])
            except IndexError: # some list is not long enough
                pass
        
        value_std.append(np.std(value_list_vertical))          
        if mean:
            value_sum.append(np.mean(value_list_vertical))
        else:
            value_sum.append(np.sum(value_list_vertical))
        
    value_sum = np.array(value_sum)
    value_std = np.array(value_std)

    return max_length, value_sum, value_std

