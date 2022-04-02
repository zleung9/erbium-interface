import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import root_scalar

### Crossing related functions ####
def find_crossings(trajectories, frames, s=0.02, threshold=0.0):
    '''
    Find out how many times each trajectory cross the threshold.
    '''
    crossings = []
    for traj in trajectories:
        traj_func = UnivariateSpline(frames, traj-threshold, s=s)             
        roots = traj_func.roots()
        crossings.append(((roots>-50)&(roots<50)).sum())
    crossings = np.array(crossings)

    return crossings


def optimize_threshold(trajectories, frames, s=0.02, threshold=0):
    '''
    Find out the value of the threshold that trajectories ensemble cross most.
    '''
     
    crossings = find_crossings(trajectories, frames, s=s, threshold=threshold)
    average_crossing = crossings.sum()/len(crossings)
    
    return average_crossing
    

def get_correlation_single(distance_array, threshold=0, ascending=True):
    """
    Find the correlation function for a single trajectory.

    Parameters
    ----------
    distance_array : array_like
        The 2-col distance arrays between Er and (the closest) Oxygen in DEHP
        for all frames. col1 is frame indices (0-indexed), the second column is
        distance values. The objected returned by `calculate_ER_OP_distance`.
    threshold : float
        The threshold of distance.
    ascending : bool
        If true, approaching the threshold from further away.

    Returns
    -------
    signs_array : array_like
        The 2-col signes array of the same shape as `distance_array`.
        Col1 is the index that signs turned 1 for the first time
        Col2 is the sign of the threshold-distance.
    """

    distance = distance_array[:,1]

    if not ascending: # flip the sign if descending
        distance = -distance
    
    traj_signs = np.zeros_like(distance)    
    traj_signs[distance <= threshold] = 1
    
    # find out the index where the signs first turns 1
    length = len(traj_signs)
    index = 0
    while not traj_signs[index]:
        index += 1
        if index == length: # never reached threshold
            index_1st_nonzero = -1 
            break
    else:
        index_1st_nonzero = index
    
    return (index_1st_nonzero, traj_signs)


def get_correlation_all(
    result,
    threshold = 0,
    ascending = True, 
    align_zero = True,
    mean = True
):
    
    """
    Return the `vertical_sum` result for all the trajectories in `result`.
    """
    
    correlation_list = []

    for trj in result:
        zero_index, signs = get_correlation_single(
            result[trj], 
            threshold = threshold, 
            ascending = ascending
        )
        if not align_zero: 
            zero_index = 0 # do not zero the index 
        if zero_index == -1: 
                continue # doesnot append if  the trajectory didn't reach the threshold
        correlation_list.append(signs[zero_index:])

    if len(correlation_list) == 0: # no trajectory reached this threshold
        return None

    max_length, correlation_sum, correlation_std = vertical_sum(correlation_list, mean=mean) 
    
    result =  np.stack(
        [
            np.arange(max_length), # the frames counting from the first non-zero index
            correlation_sum, # the averaged sign: correlation function
            correlation_std,  # the standard deviation of the correlation function.
        ], axis= 1
    )

    return result


def vertical_sum(correlation_list, mean=True):
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
    correlation_list : list
        The input list of lists (see the example above)
    mean : bool
        If true return mean, otherwise return sum.

    Returns
    -------
    max_length : int
        The maximum length of signs (15 in the above example)
    correlation_sum : list
        A list whose element is the sum/mean in each column.
    correlation_std : list
        A list whose element is the standard deviation in each column.
    """

    # maximum length of signs array used as the tota span 
    max_length = max([len(corr) for corr in correlation_list])
    
    correlation_sum = []
    correlation_std = []
    
    for i in range(max_length): # frame from the first nonzero index
        correlation_per_frame = []
        for corr in correlation_list: # get all the ith element of correlation list, if any
            try:
                correlation_per_frame.append(corr[i])
            except IndexError: # some list is not long enough
                pass
        
        correlation_std.append(np.std(correlation_per_frame))          
        if mean:
            correlation_sum.append(np.mean(correlation_per_frame))
        else:
            correlation_sum.append(np.sum(correlation_per_frame))
    
    return max_length, correlation_sum, correlation_std