import numpy as np
from numpy import random
from phi.flow import struct
from phi.physics.field import AnalyticField


def random_unit_vec(rank, batch_size):
    vec = np.ones((1, batch_size))
    for _ in range(1, rank):
        r = np.random.uniform(0, 2 * np.pi, (1, batch_size))
        vec = np.concatenate([vec * np.sin(r), np.cos(r)])
    return vec

@struct.definition()
class GaussianClash(AnalyticField):
    def __init__(self, batch_size, rank=1):
        AnalyticField.__init__(self, rank=rank)
        self.axis = random_unit_vec(rank, batch_size)
        self.posloc = (-1 * self.axis) * np.random.uniform(0.1, 0.3, batch_size) + 0.5
        self.posamp = np.random.uniform(0, 3, batch_size)
        self.possig = np.random.uniform(0.05, 0.15, batch_size)
        self.negloc = self.axis * np.random.uniform(0.1, 0.3, batch_size) + 0.5
        self.negamp = np.random.uniform(-3, 0, batch_size)
        self.negsig = np.random.uniform(0.05, 0.15, batch_size)

    def sample_at(self, idx, collapse_dimensions=True):
        idx = np.moveaxis(idx, 0, -1)  # batch last

        pos = self.posamp * np.exp(-0.5 * np.sum((idx - self.posloc) ** 2, axis=-2, keepdims=True) / self.possig ** 2)
        neg = self.negamp * np.exp(-0.5 * np.sum((idx - self.negloc) ** 2, axis=-2, keepdims=True) / self.negsig ** 2)
        amplitudes = pos + neg
        result = self.axis * amplitudes
        
        result = np.moveaxis(result, -1, 0)
        return result

    @struct.constant()
    def data(self, data):
        return data


@struct.definition()
class GaussianForce(AnalyticField):
    def __init__(self, batch_size, rank=1):
        AnalyticField.__init__(self, rank=rank)
        self.axis = random_unit_vec(rank, batch_size)
        self.loc = self.axis * np.random.uniform(-0.1, 0.1, (rank, batch_size)) + 0.5
        self.amp = np.random.uniform(-0.05, 0.05, (rank, batch_size)) * 32
        self.sig = np.random.uniform(0.1, 0.4, (rank, batch_size))

    def sample_at(self, idx, collapse_dimensions=True):
        idx = np.moveaxis(idx, 0, -1)  # batch last to match random values
        amplitudes = self.amp * np.exp(-0.5 * np.sum((idx - self.loc) ** 2, axis=-2, keepdims=True) / self.sig ** 2)
        result = self.axis * amplitudes

        result = np.moveaxis(result, -1, 0)
        return result

    @struct.constant()
    def data(self, data):
        return data