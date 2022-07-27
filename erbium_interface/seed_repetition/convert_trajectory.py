import os
import vmd 
from erbium_interface.utils import utils, trajectory


def convert_trajectory(trj_dir, dcd_dir, start=0, end=100, logger=None):
    
    if logger is not None:
        logger.info("Started!")

    trj_df = pd.DataFrame(
        data = [
            [
                *map(int, f.split('_')[-3:-1]), 
                f,
                '_'.join(f.split('_')[:-2])+'.cms',
            ]
            for f in os.listdir(trj_dir) if f.endswith('_trj')
        ],
        columns = ["seed_p", "seed_v", "trj_folder", 'cms_file']
    )

    # iterate velocity seed for 0~27
    for seed_v in np.arange(start, end): 

        indices = np.where(trj_df['seed_v'] == seed_v)[0]

        # iterate all position seeds for that velocity seed (should be 100 of them)
        for idx in indices: 

            _, _, trj_folder, cms_file, = trj_df.iloc[idx].values

            trajectory.save_trajectory(
                os.path.join(seed_dir, cms_file), # input cms file
                os.path.join(trj_dir, trj_folder, 'clickme.dtr' ), # trajectory
                output_dir = dcd_dir,
                logger = logger
            )
    if logger is not None:
        logger.info("Finished!")



if __name__ == "__main__":

    import argparse
    import pandas as pd
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed_dir', type = str, default = "./seed_files/",
                        help = "The folder for the 100 seed files")
    parser.add_argument('-t', '--trj_dir', type = str, default = None,
                        help = "The folder for Desmond trajecotiies")
    parser.add_argument('-o', '--output_dir', type = str, default = "./dcd_trajectories",
                        help = "The folder to save trajectories")
    parser.add_argument('-l', '--log_file', type = str, default = "logfile.log",
                        help = "The log file")
    parser.add_argument('-w', '--work_dir', type = str, default = None,
                        help = "The working directory where the result and log is saved.")

    args = parser.parse_args()
    log_file = args.log_file
    seed_dir = args.seed_dir
    trj_dir = args.trj_dir
    work_dir = args.work_dir
    dcd_dir = args.output_dir
    os.makedirs(dcd_dir, exist_ok=True) # gurantee work_dir exists

    # create logger
    logger = utils.create_logger("distance", os.path.join(work_dir, log_file))

    # Make directories for saved trajectores
    dcd_folder = os.path.join("./dcd_trajectories/")
    if not os.path.isdir(dcd_folder):
        os.mkdir(dcd_folder)

    # convert trajectories
    convert_trajectory(trj_dir, dcd_dir, start=51, end=80, logger=logger)
    
    
    