
import numpy as np
import pandas as pd

class Distance():
    def __init__(self, name, pair, distance):

        self.pair =  tuple(pair) # the atom ID's of the selected pair
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
            "pair": self.pair, # the atom ID's of the selected pair
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
        return cls(p.name, p.pair, p._data)

    def to_numpy(self):
        return self._data
    

class ErSPCDistance(Distance):
    
    def __init__(self, name, pair, distance):
        super().__init__(name, pair, distance)
        # never: never 1st_shell, binding: 1st_shell upon binding, ever: once 1st_shell
        self.first_shell_type = None

    @property
    def metadata(self):
        self.metadata_ = super(ErSPCDistance, self).metadata
        self.metadata_.update(
            {
                'first_shell_type': self.first_shell_type
            }
        )
        return self.metadata_
    
    def get_1st_shell_type(self, threshold=3.0, binding_frame=None):
        """Get the type of water molecules:
        "never": Never comes within the first shell
        "once": Ever comes within the first shell
        "binding": Once within the first shell between T-20 to T+10 where T is the binding frame.
        """
        is_1st_shell = (self._data <= threshold)
        if is_1st_shell.sum() == 0: # Water never goes into 1st shell
            self.first_shell_type = 'never'
        elif is_1st_shell.sum() > 0: # Water is once in the 1st shell
            self.first_shell_type = 'once'
            if binding_frame is not None:
                assert binding_frame > 10 # Water is the 1st shell upon binding
                select = (self.frames>binding_frame-20) & (self.frames<binding_frame-10)
                if (self._data[select] <= threshold).any():
                    self.first_shell_type = 'binding'
        return self.first_shell_type

    def get_frame_upon_eject(self, threshold=3.0, start=0, end=-1, stay=1):
        """Get the frame where the water first first goes beyond threshold for `stay` amount of
        frame. E.g. `threshold=3.0` and `stay=10` means get the frame where the Er-SPC goes above
        3.0A and stay above 3.0A for the next 10 frames.
        Only frames between `start` and `end` will be considered.
        """
        if end == -1:
            end = len(self.frames)

        eject_frame, stay_frames = None, 0
        frame_1st_eject, stay_1st_eject = None, None
        self._ejection_frames = []
        
        record_1st_occurance = True
        for f in self.frames[start:end]:
            if self._data[f] <= threshold or f == end-1:
                if stay_frames > 0: # At turning point, record then reset ejection.
                    self._ejection_frames.append([eject_frame, stay_frames])
                    if stay_frames >= stay and record_1st_occurance: 
                    # record the frame it stays above threshold for the first time
                        frame_1st_eject = eject_frame
                        stay_1st_eject = stay_frames
                        record_1st_occurance = False
                    stay_frames = 0
                eject_frame = f + 1 # Look at next frame if no ejection at this frame.
            else: 
                stay_frames += 1

        return frame_1st_eject, stay_1st_eject

class ErPDistance(Distance):
    
    def __init__(self, name, pair, distance):
        super().__init__(name, pair, distance)
        self.binding_frame = None # the frame where binding happens, initial value is None.

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
        """ Equivalently find the last frame that distance is bigger than threshold, after which
        the P atom will not return above threshold.
        If no binding happens, return -1.
        """
        try:
            self.binding_frame = np.where(self._data >= threshold)[0][-1]
        except:
            self.binding_frame = -1
        return self.binding_frame
    
    def get_frame_upon_arriving_at(self, threshold=4.3):
        """Equivalently find the first frame that distance is no bigger than threshold, or the frame
        where the distance **first** reaches the threshold.
        """
        return np.where(self._data <= threshold)[0][0]

class Distances():
    def __init__(self, distance_list):
        # A list of the indices of atoms for which the distance is calculated.
        self.pairs = [distance.pair for distance in distance_list]
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
        else:
            # Slicing will return the same Distances class with sliced elements
            selected_dist_list = np.array(self._distance_list)[item].tolist()
            return self.__class__(selected_dist_list)

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
    
    def get_1st_shell(self, frame=None, threshold=3):
        """
        Get a Distances object that contains waters in the first shell for a given frame.
        """
        assert frame is not None
        dist_array = self.to_numpy(start=frame, end=frame+1).reshape(-1,)
        return self.__getitem__((dist_array <= threshold))

    

class ErPDistances(Distances):
    def __init__(self, distance_list):
        super().__init__(distance_list=distance_list)