import numpy as np
import mdtraj

from erbium_interface.utils import utils, trajectory


def calculate_ER_OP_distance(mol):
    """
    Calculate the distances between Er and (closest) Oxygen in DEHP over all
    frames.
    It can be extended to calculated distances between multiple pairs.

    Parameters
    ----------
    mol : md_traj molecules object.
        The molecule to calculate distances on.
    
    Returns
    -------
    distance : array_like
        The 2-col distance arrays between Er and (the closest) Oxygen in DEHP
        for all frames. col1 is frame indices (0-indexed), the second column is
        distance values.
    """

    # Identify the Oxygen atom in DEHP that binds to Er
    top = mol.topology 
    Er = top.select("type==Er")
    ODEH = top.select("resname DEHP or resname HDEH and type==O")
    distances_Er_ODEH = mdtraj.compute_distances(mol,
                                             np.array([[Er[0],O_atom] for O_atom in ODEH ]),periodic=True)
    # select the binding oxygen index.
    closer_oxygen = np.argmin(distances_Er_ODEH[-10:].mean(axis=0)) 
    distance_Er_ODEH = distances_Er_ODEH[:,[closer_oxygen]]
    
    return np.hstack((mol.time.reshape(-1,1), distance_Er_ODEH))


    