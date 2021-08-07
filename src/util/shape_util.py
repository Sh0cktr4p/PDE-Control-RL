import numpy as np
from typing import List, Optional, Tuple
from numpy.core.fromnumeric import swapaxes
from phi.flow import struct
from phi.tf.flow import AnalyticField, Domain


class Shape:
	def __init__(self, bounds: np.ndarray, offsets: Optional[Tuple[float]]=None):
		if offsets is None:
			offsets = np.array([np.random.uniform(0 + bound, 1 - 2 * bound) for bound in bounds])
		self.offsets = offsets

	def sample_sdf(self, idx: np.ndarray) -> np.ndarray:
		pass

	@staticmethod
	def random():
		pass


class Circle(Shape):
	def __init__(self, radius: float, offsets: Optional[Tuple[float]]=None):
		self.radius = radius
		super().__init__(np.array([radius, radius]) * 2, offsets)

	def sample_sdf(self, idx: np.ndarray) -> np.ndarray:
		origin = self.offsets + self.radius
		d = np.linalg.norm(idx - origin, axis=-1, keepdims=True) - self.radius
		return d

	@staticmethod
	def random():
		return Circle(np.random.uniform(0.05, 0.3))

class Rect(Shape):
	def __init__(self, width: float, height: float, offsets: Optional[Tuple[float]]=None):
		self.dimensions = np.array([height, width])
		super().__init__(self.dimensions, offsets)

	def sample_sdf(self, idx: np.ndarray) -> np.ndarray:
		origin = self.offsets + self.dimensions / 2.0
		d = np.abs(idx - origin) - self.dimensions
		d = np.linalg.norm(np.maximum(d, 0), axis=-1, keepdims=True) + np.minimum(np.max(d, axis=-1, keepdims=True), 0)
		return np.minimum(d, 0)

	@staticmethod
	def random():
		return Rect(np.random.uniform(0.05, 0.2), np.random.uniform(0.05, 0.2))

class Diamond(Shape):
	def __init__(self, d: int):
		self.diameter = np.array([d])
		super().__init__(np.array([d, d]))

	def _sdf(self, field_shape: Tuple[int], offsets: Tuple[int]) -> np.ndarray:
		r = self.diameter / 2
		origin = offsets + r - 0.5
		i = np.indices(field_shape)
		q = np.abs(i - origin.reshape(2, 1, 1))
		h = np.clip(-2.0 * (q[0] - q[1]) * r / np.sum(r ** 2), -1, 1)
		d = np.linalg.norm(q - 0.5 * r * np.array([1.0 - h, 1.0 + h]), axis=0)
		return np.where(np.sum(q * r, axis=0) - r ** 2 > 0, -d, 0)

	def sample_sdf(self, idx: np.ndarray) -> np.ndarray:
		r = self.diameter / 2
		origin = self.offsets + r
		q = np.abs(idx - origin)
		q = np.swapaxes(q, 0, -1)
		h = np.clip(-2.0 * (q[0] - q[1]) * r / np.sum(r ** 2), -1, 1)
		d = np.linalg.norm(q - 0.5 * r * np.array([1.0 - h, 1.0 + h]), axis=0, keepdims=True)
		dist = np.where(np.sum(q * r, axis=0) - r ** 2 > 0, 0, -d)
		dist = np.swapaxes(dist, 0, -1)
		return dist

	@staticmethod
	def random():
		return Diamond(np.random.uniform(0.05, 0.3))

'''
shapes = [
	Rect(3, 3),
	Rect(3, 4), 
	Rect(4, 3), 
	Rect(5, 2), 
	Rect(2, 5), 
	Rect(6, 2), 
	Rect(2, 6),
	Diamond(4), 
	Diamond(5), 
	Diamond(6)
]
'''

@struct.definition()
class ShapeField(AnalyticField):
	def __init__(self, batch_size: int, total_density: float, shape_list: Optional[List[Shape]]=None):
		AnalyticField.__init__(self, rank=2)
		if shape_list is None:
			shape_list = [
				Circle,
				Rect,
				Diamond,
			]

		self.shapes = [shape_list[np.random.randint(len(shape_list))].random() for _ in range(batch_size)]

	def sample_at(self, idx: np.ndarray, collapse_dimensions:bool=True) -> np.ndarray:
		sdf = np.array([shape.sample_sdf(idx[0]) for i, shape in enumerate(self.shapes)])
		density = np.where(sdf < 0, 1, 0)
		return density

	@struct.constant()
	def data(self, data):
		return data