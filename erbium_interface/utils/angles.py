import numpy as np
import pandas as pd
from erbium_interface.utils.distance import Distance, Distances


class Angle():
    def __init__(self, traj, pair1, pair2):
        self.indices = [None, None, None] # follow the order of a--o--b
        for _id in pair1:
            if _id not in pair2:
                self.indices[0] = _id
            else:
                self.indices[1] = _id
        for _id in pair2:
            if _id not in pair1:
                self.indices[2] = _id
        self.coords = [traj.trajectory.xyz[:, i, :] for i in self.indices]


    @property
    def cosine(self):
        self.vectors = [
            self.coords[0] - self.coords[1],
            self.coords[2] - self.coords[1]
        ]
        self.distances = [np.linalg.norm(v, axis=1) for v in self.vectors]

        _cosine =  (self.vectors[0] * self.vectors[1]).sum(axis=1).T \
                / (self.distances[0] * self.distances[1])
        return _cosine

    @property
    def angle(self,  degree=True):
        a = np.arccos(self.cosine)
        if degree:
            a = a / np.pi * 180
        return a

