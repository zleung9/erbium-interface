import os
import pickle
import copy

import pandas as pd

import vmd 
import mdtraj


def get_trajectories(dcd_dir):
    """
    Parse the trajectory folder that contains .dcd, .pdb and .mae files.
    Returns
    -------
    trajectory_list : List[str]
        A list trajectory names. e.g. ["md_20220308_1_3", "md_20220308_1_2", ...]
        The corresponding topology or trajectory is: name.pdb or namd.dcd, respectively.
    """
    trajectory_list = []
    
    for f in os.listdir(dcd_dir):
        if f.endswith('.dcd'):
            prefix = f.replace('.dcd', '')
            if not os.path.isfile(os.path.join(dcd_dir, prefix + '.pdb')):
                continue
        
            trajectory_list.append(prefix)
    
    return trajectory_list


def parse_trajectories(
    call,  # 
    dcd_dir = None,
    selected_names = None,
    logger = None,
    update = None
):
    """
    Parameters
    ----------
    call : python callable
        A function that takes in a molecule (loaded with "md_traj") and 
        produces a python object (list, dictionary, numpy array) that can be 
        either saved to a file or cnoverted to a dataframe.
    dcd_dir : str
        The absolute path for the folder that contains all the topology 
        (.mae and .pdb) and trajecotry (.dcd) files
    selected_names : list[str]
        A list of file names to be processed. Names are suffix free (e.g., do
        not contain .pdb, .mae. or.dcd). If None, process all the files in 
        `dcd_dir`.
    to_update : str
        The file to which new results are updated. It can be a dataframe,
        a python pickle file (dictionary). If not specified, usually a series
        of data files are written to `output_dir`.
    """
    
    if dcd_dir is None:
        dcd_dir = os.path.dirname(__file__)
    assert os.path.isdir(dcd_dir)
    
    # define the result
    trajectory_list = get_trajectories(dcd_dir)
    if selected_names is not None:
        trajectory_list = [trj for trj in trajectory_list  if trj in selected_names]
    
    # produce the result for each trajectory
    result = {trj: None for trj in trajectory_list}
    for trj in trajectory_list:
        mol = load_molecule(
                os.path.join(dcd_dir, trj + '.pdb'),
                os.path.join(dcd_dir, trj + '.dcd'),
                method = 'md_traj'
            )
        try: 
            result[trj] = call(mol)
            if logger is not None:
                logger.info(f"'{call.__name__}' applied on '{trj}' successfully! ")
        except Exception:
            if logger is not None:
                logger.error(f"Failed --> {Exception}.")
    
    if update is None:
        result_new = result
    else:
        assert os.path.isfile(update) # make sure the file to update exists.
        result_new = update_result(load_result(update), result)
    
       
    return result_new


def load_result(result_path, folder_loader=None):
    """
    a function for loading result.
    """
    if result_path.endswith(".csv"):
        result = pd.read_csv(result_path)
    elif result_path.endswith(".pkl"):
        result = pickle.load(open(result_path, 'rb'))
    elif os.path.isdir(result_path):
        result = folder_loader(result_path)
    
    return result


def update_result(old_result, incoming):
    """
    So far only support updating dictionary.
    """

    new_result = copy.copy(old_result)
    for key, value in incoming.items():
        new_result[key] = value
    
    return new_result


def load_molecule(topology, trajectory, method='vmd'):
    '''
    Load molecule using vmd or md_traj.
    '''
    if method == 'vmd':
        mol = vmd.molecule.load("mae", topology)
        vmd.molecule.read(mol, "dtr", trajectory,waitfor = -1)
        vmd.molecule.delframe(mol,0,0,0) # The ".cms" file only provide topology, not positions.
    elif method == 'md_traj':
        mol = mdtraj.load(trajectory, top = topology)
        mol.xyz *= 10 # convert coordinates from nm to A
        mol.unitcell_lengths *= 10 # convert box dimensions from nm to A
    return mol

