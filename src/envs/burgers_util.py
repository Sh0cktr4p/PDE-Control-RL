import numpy as np
from numpy import random
from phi.flow import struct
from phi.physics.field import AnalyticField


def random_positive_unit_vec(rank, batch_size):
    vec = np.ones((1, batch_size))
    for _ in range(1, rank):
        r = np.random.uniform(0, np.pi / 2, (1, batch_size))
        vec = np.concatenate([vec * np.sin(r), np.cos(r)])
    return vec

@struct.definition()
class GaussianClash(AnalyticField):
    def __init__(self, batch_size, rank=1, vector_valued=False):
        AnalyticField.__init__(self, rank=rank)
        #self.base_elem = random_positive_unit_vec(rank if vector_valued else 1, batch_size)
        self.axis = random_positive_unit_vec(rank, batch_size)

        self.posloc = (-1 * self.axis) * np.random.uniform(0.1, 0.3, batch_size) + 0.5
        self.posamp = np.random.uniform(0, 3, batch_size)
        self.possig = np.random.uniform(0.05, 0.15, batch_size)
        self.negloc = self.axis * np.random.uniform(0.1, 0.3, batch_size) + 0.5
        self.negamp = np.random.uniform(-3, 0, batch_size)
        self.negsig = np.random.uniform(0.05, 0.15, batch_size)

    def sample_at(self, idx, collapse_dimensions=True):
        idx = np.moveaxis(idx, 0, -1)  # batch last

        # idx: (128, 128, 2, 10)
        # loc: (2, 10)
        # pos: (128, 128, 2, 10)
        x = np.sum((idx - self.posloc) ** 2, axis=-2, keepdims=True)
        pos = self.posamp * np.exp(-0.5 * x / self.possig ** 2)
        print(x.shape)
        neg = self.negamp * np.exp(-0.5 * np.sum((idx - self.negloc) ** 2, axis=-2, keepdims=True) / self.negsig ** 2)
        amplitudes = pos + neg
        dir = self.axis
        #dir = 0.5 - idx
        #print(dir.shape)
        #print(amplitudes.shape)
        #dir /= np.sum(dir ** 2, axis=-2, keepdims=True)
        result = dir * amplitudes
        #result = amplitudes * np.sum((idx - 0.5) ** 2)
        #result = idx - 0.5#, axis=-2, keepdims=True))
        
        result = np.moveaxis(result, -1, 0)
        return result

    @struct.constant()
    def data(self, data):
        return data


@struct.definition()
class GaussianForce(AnalyticField):
    def __init__(self, batch_size, rank=2, vector_valued=True):
        AnalyticField.__init__(self, rank=rank)
        self.loc = np.random.uniform(0.4, 0.6, (rank, batch_size))
        self.amp = np.random.uniform(-0.05, 0.05, (rank, batch_size)) * 32
        self.sig = np.random.uniform(0.1, 0.4, (rank, batch_size))
        self.base_elem = random_positive_unit_vec(rank if vector_valued else 1, batch_size)

    def sample_at(self, idx, collapse_dimensions=True):
        idx = np.moveaxis(idx, 0, -1)  # batch last to match random values
        amplitudes = np.prod(self.amp * np.exp(-0.5 * (idx - self.loc) ** 2 / self.sig ** 2), axis=-2, keepdims=True)
        result = amplitudes * self.base_elem
        result = np.moveaxis(result, -1, 0)
        return result

    @struct.constant()
    def data(self, data):
        return data