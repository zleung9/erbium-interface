
import numpy as np
import pandas as pd

class Distance():
    def __init__(self, name, pair, distance):

        self.atom_id =  tuple(pair) # the atom ID's of the selected pair
        self.atom_name = (None, None) # the atom names of the selected pair
        self.name = name # the assigned name of this Distance object
        self.time = [] # what is this?
        self.ps_per_frame = 1 # how many ps is represented by each frame.
        self.frames = np.arange(len(distance))
        self._data = distance
        self._metadata = {} 
   
    def __getitem__(self, item):
        return self._data[item]

    @property
    def metadata(self):
        self._metadata = {
            "atom_id": self.atom_id, # the atom ID's of the selected pair
            "atom_name": self.atom_name, # the atom names of the selected pair
            "name": self.name, # the assigned name of this Distance object
            "time": self.time, # what is this?
            "ps_per_frame": self.ps_per_frame # how many ps is represented by each frame.
        }
        return self._metadata
    
    @property
    def data(self):
        return self._data
        
    @classmethod
    def from_parent(cls, p):
        """ Construct an child object from an existing Distance instance `p`.
        """
        return cls(p.name, p.atom_id, p._data)

    def to_numpy(self):
        return self._data
    

class ErSPCDistance(Distance):
    
    def __init__(self, name, pair, distance):
        super().__init__(name, pair, distance)
        # never: never 1st_shell, binding: 1st_shell upon binding, ever: once 1st_shell
        self.type_1st_shell_ = None

    @property
    def metadata(self):
        self.metadata_ = super(ErSPCDistance, self).metadata
        self.metadata_.update(
            {
                'type_1st_shell': self.type_1st_shell_
            }
        )
        return self.metadata_
    
    def get_1st_shell_type(self, threshold=3.0, binding_frame=None):
        is_1st_shell = (self._data <= threshold)
        if is_1st_shell.sum() == 0: # Water never goes into 1st shell
            self.type_1st_shell_ = 'never'
        elif is_1st_shell.sum() > 0: # Water is once in the 1st shell
            self.type_1st_shell_ = 'once'
            if binding_frame is not None:
                assert binding_frame > 10 # Water is the 1st shell upon binding
                select = (self.frames>binding_frame-20) & (self.frames<binding_frame-10)
                if (self._data[select] <= threshold).sum() > 0:
                    self.type_1st_shell_ = 'binding'
        return self.type_1st_shell_


class ErPDistance(Distance):
    
    def __init__(self, name, pair, distance):
        super().__init__(name, pair, distance)
        self.binding_frame = None # the frame where binding happens

    @property
    def metadata(self):
        self.metadata_ = super(ErPDistance, self).metadata
        self.metadata_.update(
            {
                'binding_frame': self.binding_frame
            }
        )
        return self.metadata_

    def get_frame_upon_binding(self, threshold=4.3):
        """ Equivalently find the last frame that distance is bigger than threshold.
        """
        self.binding_frame = np.where(self._data >= threshold)[0][-1]
        return self.binding_frame
    
    def get_frame_upon_arriving_at(self, threshold=4.3):
        """Calculate the frame number where the distance first reaches threshold.
        """
        return np.where(self._data <= threshold)[0][0]

class Distances():
    def __init__(self, distance_list):
        # A list of the indices of atoms for which the distance is calculated.
        self.pairs = [distance.atom_id for distance in distance_list]
        self.num_pairs = len(self.pairs)
        # A list of Distance object
        self._distance_list = distance_list 
        # Use the firs frame for all, because they are meant to be the same
        self.frames = distance_list[0].frames 
        self.num_frame = len(self.frames)

    def __getitem__(self, item):
        """Query Distance object through its index, or name, or tuple of indices of the atom pair.
        """
        if isinstance(item, int):
            return self._distance_list[item]
        elif isinstance(item, str):
            raise NotImplementedError
        elif isinstance(item, tuple):
            return self._distance_list[self.pairs.index(item)]

    def __len__(self):
        return len(self._distance_list)

    @classmethod
    def from_parent(cls, p, child_cls):
        """ Construct an child_cls object from an existing Distances instance `p`.
        The `child_cls` inherits from Distances.
        """
        distance_list = [child_cls.from_parent(distance) for distance in p]
        return cls(distance_list)

    def to_list(self):
        return self._distance_list

    def to_numpy(self, start=None, end=None, smooth=0):
        if start is None:
            start = 0
        if end is None:
            end = self.num_frame
        data =  np.stack(
            [distance.data[start:end] for distance in self._distance_list]
        )
        assert data.shape[0] == self.num_pairs
        assert data.shape[1] == end - start
        return data

    def to_dataframe(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = self.num_frame

        df = pd.DataFrame(
            data = self.to_numpy(start=start, end=end),
            columns = [f"FR{int(f):04d}" for f in self.frames[start:end]]
        )
        return df

    def reaction_flux(self, threshold=None, direction='out'):
        """
        The reaction flux correlation is calculated as the following: 
        For each `Distance` in `Distances`, calculate 
        Parameters
        ----------
        direction : str
            If "out", assign frames where the distance is larger than the threshold to be 1, other 
            frames to be 0. If "in" assign frames where the distance is smaller than the threshold 
            to be 1, other frames to be 0.
        Returns
        -------
        The reaction flux coorelation
        """
        assert not threshold is None
        pass

class ErSPCDistances(Distances):
    
    def __init__(self, distance_list):
        super().__init__( distance_list=distance_list)

    def get_cn(self, start=None, end=None, threshold=3):
        dist_array = self.to_numpy(start=start, end=end)
        cn = (dist_array <= threshold).sum(axis=0) # coordination number
        return cn
    
    def get_1st_shell_upon_binding(self, binding_frame):
        pairs_1st_shell = []
        distances_1st_shell = []
        for distance in self._distance_list:
            distance.get_1st_shell_type(binding_frame=binding_frame)
            if distance.type_1st_shell_ == 'binding':
                pairs_1st_shell.append(distance.atom_id)
                distances_1st_shell.append(distance)
        return ErSPCDistances(distances_1st_shell)
    

class ErPDistances(Distances):
    def __init__(self, distance_list):
        super().__init__(distance_list=distance_list)