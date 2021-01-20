from phi.flow import struct
from phi.physics.field import AnalyticField
import numpy as np

@struct.definition()
class GaussianClash(AnalyticField):

    def __init__(self, batch_size):
        AnalyticField.__init__(self, rank=1)
        self.batch_size = batch_size

    def sample_at(self, idx, collapse_dimensions=True):
        leftloc = np.random.uniform(0.2, 0.4, self.batch_size)
        leftamp = np.random.uniform(0, 3, self.batch_size)
        leftsig = np.random.uniform(0.05, 0.15, self.batch_size)
        rightloc = np.random.uniform(0.6, 0.8, self.batch_size)
        rightamp = np.random.uniform(-3, 0, self.batch_size)
        rightsig = np.random.uniform(0.05, 0.15, self.batch_size)
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        left = leftamp * np.exp(-0.5 * (idx - leftloc) ** 2 / leftsig ** 2)
        right = rightamp * np.exp(-0.5 * (idx - rightloc) ** 2 / rightsig ** 2)
        result = left + right
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data):
        return data


@struct.definition()
class GaussianForce(AnalyticField):
    def __init__(self, batch_size):
        AnalyticField.__init__(self, rank=1)
        self.loc = np.random.uniform(0.4, 0.6, batch_size)
        self.amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
        self.sig = np.random.uniform(0.1, 0.4, batch_size)

    def sample_at(self, idx, collapse_dimensions=True):
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        result = self.amp * np.exp(-0.5 * (idx - self.loc) ** 2 / self.sig ** 2)
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data):
        return data