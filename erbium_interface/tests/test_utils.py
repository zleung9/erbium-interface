import os
from erbium_interface.utils import  trajectory

class Test_utils():

    def test_get_trajectories(self):
        dcd_folder = os.path.join(os.path.dirname(__file__), "../data/dcd_trajectories/")
        trajectory_list = trajectory.get_trajectories(dcd_folder)
        assert len(trajectory_list) == 3
        assert 'md_20220308_1_3' in trajectory_list
        assert 'md_20220308_1_1' in trajectory_list
        assert 'md_20220308_1_2' in trajectory_list


if __name__ == "__main__":
    Test_utils().test_get_trajectories()