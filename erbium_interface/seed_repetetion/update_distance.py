import numpy as np
import pandas as pd
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


def to_dataframe(result):

    """
    Convert the result dictionary to dataframe.
    """
    data = []
    for trj, dist in result.items():
        p_seed, v_seed = map(int, trj.split('_')[-2:])
        data.append(
            [p_seed, v_seed, trj, *list(dist[:,1])]
        )
    else:
        frames = [f"FR{int(f):04d}" for f in dist[:,0]]

    df = pd.DataFrame(
        data = data,
        columns = ["p_seed", "v_seed", "trajectory", *frames]
    )
    
    return df


def main():
    
    import os
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--dcd_dir', type = str, default = None,
                        help = "The folder for the 100 seed files")
    parser.add_argument('-o', '--output', type = str, default = "Er_Op_distance.csv",
                        help = "The output csv filename of the O(P)-Er distance.")
    parser.add_argument('-l', '--log_file', type = str, default = "update_distance.log",
                        help = "The log file")
    parser.add_argument('-w', '--work_dir', type = str, default = None,
                        help = "The working directory where the result and log is saved.")

    args = parser.parse_args()
    output_file = args.output
    log_file = args.log_file
    dcd_dir = args.dcd_dir
    work_dir = args.work_dir
    os.makedirs(work_dir, exist_ok=True) # gurantee work_dir exists

    # create logger
    logger = utils.create_logger("distance", os.path.join(work_dir, log_file))
    
    # get trajectories
    trajectory_list = trajectory.get_trajectories(dcd_dir)

    # calculate distances
    result = trajectory.parse_trajectories(
        calculate_ER_OP_distance,
        dcd_dir = dcd_dir,
        selected_names = trajectory_list,
        logger = logger
    )

    df = to_dataframe(result)
    df.to_csv(os.path.join(work_dir, output_file), index = False)

    # optional: save the dictionary result (for udpate in the future)
    with open(os.path.join(work_dir, output_file.replace('.csv', '.pkl')), 'wb') as f:
            pickle.dump(result, f)

if __name__ == "__main__":
    main() 