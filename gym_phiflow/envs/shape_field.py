import numpy as np
from gym_phiflow.envs import util


def print_field(field):
	s = ''
	for row in field:
		for col in row:
			s += '. ' if col == 0 else 'X '
		s += '\n'

	print(s)	

def pf(field):
	s = ''
	for row in field:
		for col in row:
			s += '%5.2f, ' % col
		s += '\n'
	print(s)

def get_shape_field_gen(shape):
	base_field = np.zeros(shape)


def sdf_square_old(field, ox, oy, w, h):
	i = np.indices(field.shape)
	dx = np.abs(i[0] - ox) - w / 2.0
	dy = np.abs(i[1] - oy) - h / 2.0
	d = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2) + np.minimum(np.maximum(dx, dy), 0)
	return np.minimum(-d, 0)


def sdf(f, o, d):
	i = np.indices(f.shape)
	d = np.abs(i - o.reshape(2, 1, 1)) - d.reshape(2, 1, 1) / 2.0
	d = np.linalg.norm(np.maximum(d, 0), axis=0) + np.minimum(np.max(d, axis=0), 0)
	return np.minimum(-d, 0)


shape_index = 0


class Shape:
	def __init__(self, size):
		self.size = size

	def sdf(self, field_shape, offsets):
		pass

	def get_sdf_field(self, field_shape, off_vals=None):
		if off_vals is None:
			ranges = [field_shape[i] - self.size[i] + 1 for i in range(len(field_shape))]
			off_vals = np.array([np.random.randint(r) for r in ranges])

		sdf_field = self.sdf(field_shape, off_vals)
		return sdf_field


class Rect(Shape):
	def __init__(self, width, height):
		self.dimensions = np.array([height, width])
		super().__init__(self.dimensions)

	def sdf(self, field_shape, offsets):
		origin = offsets + self.dimensions / 2.0 - 0.5
		i = np.indices(field_shape)
		d = np.abs(i - origin.reshape(2, 1, 1)) - self.dimensions.reshape(2, 1, 1) / 2.0
		d = np.linalg.norm(np.maximum(d, 0), axis=0) + np.minimum(np.max(d, axis=0), 0)
		return np.minimum(-d, 0)


class Diamond(Shape):
	def __init__(self, d):
		self.diameter = np.array([d])
		super().__init__(np.array([d, d]))

	def sdf(self, field_shape, offsets):
		r = self.diameter / 2
		origin = offsets + r - 0.5
		i = np.indices(field_shape)
		q = np.abs(i - origin.reshape(2, 1, 1))
		h = np.clip(-2.0 * (q[0] - q[1]) * r / np.sum(r ** 2), -1, 1)
		d = np.linalg.norm(q - 0.5 * r * np.array([1.0 - h, 1.0 + h]), axis=0)
		return np.where(np.sum(q * r, axis=0) - r ** 2 > 0, -d, 0)


shapes_basic = [Rect(2, 2)]

shapes = [Rect(3, 3), Rect(3, 4), Rect(4, 3), Rect(5, 2), Rect(2, 5), Rect(6, 2), Rect(2, 6),
		Diamond(4), Diamond(5), Diamond(6)]


field_dict = {}

def get_random_sdf_field(field_shape, shape_list=shapes, squared=False):
	return shape_list[np.random.randint(len(shape_list))].get_sdf_field(field_shape) ** (2 if squared else 1)


def to_density_field(sdf_field, total_density):
	density = np.where(np.abs(sdf_field) < 0.001, 1, 0)
	return density * total_density / np.sum(density)