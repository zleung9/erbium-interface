import os
from turtle import update
from erbium_interface.seed_repetetion import update_distance
from erbium_interface.utils import trajectory

class Test_seed_repetition():

    def test_calculate_ER_OP_distance(self):
        dcd_folder = os.path.join(os.path.dirname(__file__), "../data/dcd_trajectories/")
        
        trajectory_list = trajectory.get_trajectories(dcd_folder)
        for trj in trajectory_list:
            mol = trajectory.load_molecule(
                os.path.join(dcd_folder, trj + '.pdb'),
                os.path.join(dcd_folder, trj + '.dcd'),
                method = 'md_traj'
            )
            distances = update_distance.calculate_ER_OP_distance(mol)
            assert distances.shape == (4002, 2)

    def test_get_trajectories(self):
        dcd_folder = os.path.join(os.path.dirname(__file__), "../data/dcd_trajectories/")
        trajectory_list = trajectory.get_trajectories(dcd_folder)
        assert len(trajectory_list) == 3
        assert 'md_20220308_1_3' in trajectory_list
        assert 'md_20220308_1_1' in trajectory_list
        assert 'md_20220308_1_2' in trajectory_list


    def test_to_dataframe(self):
        dcd_dir = os.path.join(os.path.dirname(__file__), "../data/dcd_trajectories/")
        result = trajectory.parse_trajectories(
            update_distance.calculate_ER_OP_distance,
            dcd_dir = dcd_dir
        )
        update_distance.to_dataframe(result)


if __name__ == "__main__":
    Test_seed_repetition().test_to_dataframe()