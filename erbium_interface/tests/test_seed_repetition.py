import os
import numpy as np
from turtle import update
from erbium_interface.seed_repetition import update_distance
from erbium_interface.utils import trajectory

class Test_seed_repetition():
    
    dcd_dir = os.path.join(os.path.dirname(__file__), "../data/dcd_trajectories/")


    def test_calculate_ER_OP_distance(self):
        
        trajectory_list = trajectory.get_trajectories(self.dcd_dir)
        for trj in trajectory_list:
            mol = trajectory.load_molecule(
                os.path.join(self.dcd_dir, trj + '.pdb'),
                os.path.join(self.dcd_dir, trj + '.dcd'),
                method = 'md_traj'
            )
            distances = update_distance.calculate_ER_OP_distance(mol)
            assert distances.shape == (4002, 2)
        

    def test_calculate_ER_P_distance(self):
        
        trajectory_list = trajectory.get_trajectories(self.dcd_dir)
        for trj in trajectory_list:
            mol = trajectory.load_molecule(
                os.path.join(self.dcd_dir, trj + '.pdb'),
                os.path.join(self.dcd_dir, trj + '.dcd'),
                method = 'md_traj'
            )
            distances = update_distance.calculate_ER_P_distance(mol)
            assert distances.shape == (4002, 2)


    def test_to_dataframe(self):
        result = trajectory.parse_trajectories(
            update_distance.calculate_ER_OP_distance,
            dcd_dir = self.dcd_dir
        )
        update_distance.to_dataframe(result)


if __name__ == "__main__":
    Test_seed_repetition().test_get_correlation_all()