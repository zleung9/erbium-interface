import os
from re import A
import numpy as np
from erbium_interface.utils import  trajectory, binding
from erbium_interface.seed_repetition import update_distance


class Test_trajectory():

    def test_get_trajectories(self):
        dcd_folder = os.path.join(os.path.dirname(__file__), "./test_data/dcd_trajectories/")
        trajectory_list = trajectory.get_trajectories(dcd_folder)
        assert len(trajectory_list) == 5
        assert 'md_20220308_1_1' in trajectory_list
        assert 'md_20220308_1_2' in trajectory_list
        assert 'md_20220308_1_3' in trajectory_list
        assert 'md_20220308_1_4' in trajectory_list
        assert 'md_20220308_1_5' in trajectory_list
    
    def test_load_Trajectory_instance_from_file(self):
        pass



class Test_binding():
    
    dcd_dir = os.path.join(os.path.dirname(__file__), "./test_data/dcd_trajectories/")
    result_path = os.path.join(os.path.dirname(__file__), "./test_data/distance_Er_OP.pkl")


    def test_get_correlation_single(self):
        
        result = trajectory.parse_trajectories(
            update_distance.calculate_ER_OP_distance,
            dcd_dir = self.dcd_dir
        )
        nonzero_frame, signs = binding.get_correlation_single(
            result['md_20220308_1_1'], threshold=2.3, ascending=True
        )
        np.testing.assert_equal(nonzero_frame, 1174)
        np.testing.assert_equal(signs.sum(), 2817.0)
    

    def test_get_correlation_all(self):
        
        import pickle
        result = pickle.load(open(self.result_path, 'rb'))
        
        # no trajectory has reached 2.0
        correlations = binding.get_correlation_all(result, threshold = 2.07, mean=False)
        print(correlations[:,1].max())
        # np.testing.assert_equal(correlations, None)
    

    def test_vertical_sum(self):
        signs = np.array(
            [
                [1, 0, 0, 0, 0, 0, 1, 0, 1, 1],
                [0, 1, 0, 1, 1, 1, 1, 0, 1],
                [1, 1, 0, 1, 0, 1, 1],
                [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 1, 0, 1],
                [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,],
            ]
        )
        max_length, corr_sum, corr_std = binding.vertical_sum(signs, mean=False)
        assert max_length == 11       
        np.testing.assert_array_equal(corr_sum, [4,2,2,3,3,4,5,1,4,1,0])
        np.testing.assert_allclose(
            corr_std, 
            [0.47,0.47,0.47,0.5,0.5,0.47,0.37,0.4,0.4,0.47,0], 
            atol=1e-2
        )
        _, corr_mean,_ = binding.vertical_sum(signs, mean=True)
        np.testing.assert_allclose(
            corr_mean, 
            [0.66,0.33,0.33,0.5,0.5,0.66,0.83,0.2,0.8,0.33,0.0], 
            atol=1e-2
        )


if __name__ == "__main__":
    Test_binding().test_get_correlation_all()