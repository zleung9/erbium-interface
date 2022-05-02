import os
import pickle
import copy

import numpy as np
import pandas as pd

import vmd 
import mdtraj


class Distance():
    def __init__(self):
        self.atom_id = (None, None)
        self.atom_name = (None, None)
        self.name = None
        self.time = []
        self.distance = []
        self.ps_per_frame = 1

class Distances():
    def __init__(self):
        self.atom_id = (None, None)
        self.name_list = []
        self.distance_list = []
    def to_numpy(self):
        pass

class Trajectory():
    """
    An trajectory object read from dtr/dcd. It can be saved to dcd/pdb pairs.

    """
    def __init__(self, top_path, trj_path, method, load=False):
        self.top_path = top_path
        self.trj_path = trj_path
        self._method = method
        self._raw = True
        self.name = None
        self.molecule = None
        
        if load:
            self.load_molecule(self.top_path, self.trj_path, method = self._method)
    
    @classmethod
    def from_dtr(cls, top_path, trj_path):
        return Trajectory(top_path, trj_path, 'vmd', load=True)

    @classmethod
    def from_dcd(cls, top_path, trj_path):
        return Trajectory(top_path, trj_path, 'md_traj', load=True)


    def save_dcd(self, logger=None):
        """Save Trajectory as dcd/pdb/mae format.
        """
        assert self._method == 'vmd', "Convert Desmond trajectories only"

        sel_keep = reduce_selection(self.molecule)
        try:
            vmd.molecule.write(
                self.molecule, "mae", 
                self.trj_path.replace('_trj','.mae'),
                selection=sel_keep,last=0
            )
            vmd.molecule.write(
                self.molecule,"pdb", 
                self.trj_path.replace('_trj','.pdb'), 
                selection=sel_keep,last=0
            )
            vmd.molecule.write(
                self.molecule,"dcd", 
                self.trj_path.replace('_trj','.dcd'),
                selection=sel_keep
            ) 
            vmd.molecule.delete(self.molecule)
            if logger is not None:
                logger.info(f"{self.name}:\tTrajectory saved to dcd!")
        except:
            if logger is not None:
                logger.error(f"{self.name}:\tSaving trajectory to dcd failed!")
            vmd.molecule.delete(self.molecule)
            raise

       
    def load_molecule(self, topology, trajectory, method='vmd'):
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

        self.molecule = mol
    
    def delete_molecule(self):
        del self.molecule
        self.molecule = None


    def get_distance(self):
        """Return a Distance object.
        """
        return Distance()


class Trajectories():

    def __init__(self, name_list=[], trj_list=[]):
        self.name_list = name_list
        self.trj_list = trj_list
    
    @classmethod
    def from_folder(cls, trj_dir):
        trj_list = []
        name_list = []
        for path_ in os.listdir(trj_dir):
            if path_.endswith('.dcd'):
                trj_path = path_
                top_path = path_.replace('.dcd', '.pdb')
                if not os.path.isfile(os.path.join(top_path)):
                    continue

                trj_list.append(
                    Trajectory.from_dcd(top_path, trj_path)
                )
            
        return Trajectories(name_list, trj_list)
    
    def get_distances(self):
        for trj in self.trj_list:
            trj.get_distance

        


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
    update : str
        The file to which new results are updated. It can be a dataframe,
        a python pickle file (dictionary). If not specified, usually a series
        of data files are written to `output_dir`.
    """
    
    if logger is not None:
        logger.info("Started!")

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

    if logger is not None:
        logger.info("Finished!")

    return result_new


def load_result(result_path, folder_loader=None):
    """
    A function for loading result.
    
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


def reduce_selection(mol, threshold=5.5):
    '''
    Reduce selection to: Er ion, HDEHP, water molecules within 5.5 A of Er, Cl ion
    '''
    num_frames = vmd.molecule.numframes(mol)

    # get coordinates for Er and H2O
    sel_Er = vmd.atomsel("element Er",molid=mol,frame=0)
    coord_Er = get_coordinates(sel_Er,molid=mol)
    sel_OW = vmd.atomsel("resname SPC and element O", molid=mol, frame=0)
    coord_OW = get_coordinates(sel_OW,molid=mol)

    # calculate distances between Er and all water oxygens (corrected for pierodic boundary)
    dist_Er_OW = np.linalg.norm(get_displacement(coord_Er[:,0,:], coord_OW), axis=2)

    # select the water molecule that is at least once closer than 5.5A to Er
    select = (dist_Er_OW <= 5.5).any(axis=0)
    selected_resid = np.array(sel_OW.resid)[select]
    
    # Keep Er, DEHP and selected waters for dcd trajectory
    select_string = " or ".join(["element Er", 
                                 "element Cl", 
                                 "resname HDEH", 
                                 *[f"resid {i}" for i in selected_resid]])
    sel_keep = vmd.atomsel(select_string,molid=mol, frame=0)
    
    return sel_keep


def get_coordinates(sel,molid=-1):
    '''
    Calculate the distance between the center of mass of "seltext1" and each member in "seltext2".
    Default selection is made on top molecule (molid=-1) and last frame (frame=-1).
    '''
    # create the container for coordinates with dimensions:[total_frames,total atoms, (x,y,z)]
    num_frames = vmd.molecule.numframes(molid)
    coordinates = np.zeros((num_frames,len(sel),3))
    for frame in range(num_frames):
        sel.frame = frame 
        coordinates[frame] = np.array([sel.x,sel.y,sel.z]).transpose()
    return coordinates


def get_displacement(com_A, coord_B,molid=-1):
    '''
    Calculate the coordinate differences with periodic boundary. 

    Input:
    ------
    com_A: shape=(number of frames,3), xyz coordinates for the Er ion
    coord_B: shape=(number of frames, number of selected atoms, 3), xyz coordinates for selected atoms.
    
    Output:
    ------
    displace: shape=coord_B.shape, 3-D displacement vector of coord_B w.r.t. com_A.
    '''

    volumetric = vmd.molecule.get_periodic(molid)
    volume = [volumetric['a'], volumetric['b'], volumetric['c']]
    vector = np.zeros(coord_B.shape)
    for i in range(vector.shape[1]): # loop over atoms
        for j in range(3): # loop over x,y,z
            vector[:,i,j] = coord_B[:,i,j]-com_A[:,j] # calculate displacement
            off = abs(vector[:,i,j]) > volume[j]/2
            if np.sum(off) == 0: continue # no need to correct
            correction = (abs(vector[:,i,j][off])-volume[j])*np.sign(vector[:,i,j][off])
            vector[:,i,j][off] = correction
    return vector