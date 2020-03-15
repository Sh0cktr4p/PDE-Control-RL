import numpy as np
from gym_phiflow.envs import util


def print_field(field):
	s = ''
	for row in field:
		for col in row:
			s += '. ' if col == 0 else 'X '
		s += '\n'

	print(s)	


def get_shape_field_gen(shape):
	base_field = np.zeros(shape)


class Shape:
	def __init__(self, indices):
		size = [max(l) - min(l) + 1 for l in indices]
		self.indices = indices
		self.size = size

	def rand_off_indices(self, field_shape):
		ranges = [field_shape[i] - self.size[i] + 1 for i in range(len(field_shape))]
		offsets = np.tile([np.random.randint(r) for r in ranges], (len(self.indices[0]), 1)).T
		return tuple(self.indices + offsets)

	def get_field(self, field_shape):
		field = np.zeros(field_shape, dtype=np.float32)
		field[self.rand_off_indices(field_shape)] = 1 / len(self.indices[0])
		return field


class Rect(Shape):
	def __init__(self, width, height):
		indices = (np.repeat(np.arange(height), width), np.tile(np.arange(width), height))
		super().__init__(indices)


class Diamond(Shape):
	def __init__(self, d):
		rep_c = (-2 * abs(np.arange(d) - d / 2 + 0.5) + d + (1 - d % 2)).astype(np.int32)
		xs = np.repeat(np.arange(d), rep_c)
		ys = np.concatenate([np.arange(d // 2 - i // 2, d // 2 + i // 2 + i % 2) for i in rep_c])
		super().__init__((xs, ys))



def get_random_field(field_shape):
	shapes = [Rect(3, 3), Rect(3, 4), Rect(4, 3), Rect(5, 2), Rect(2, 5), Rect(6, 2), Rect(2, 6),
			Diamond(4), Diamond(5), Diamond(6)]
	return shapes[np.random.randint(len(shapes))].get_field(field_shape)