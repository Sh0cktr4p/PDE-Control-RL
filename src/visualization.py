import os
import numpy as np
import phi.flow as phiflow
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import imageio


def field_to_channel(field, max_value, signed=False):
	obs = field
		
	assert obs.shape[-1] < 3, "3D visualization not (yet) supported"

	height, width = field.shape[-3:-1]

	# Visualization should display field vector length
	obs = np.linalg.norm(obs, axis=-1).reshape(height, width)

	# Get the color values
	if signed:
		return np.clip((obs + max_value) / (2.0 * max_value), 0.0, 1.0)
	else:
		return np.clip(obs / max_value, 0.0, 1.0)


def single_field_to_rgb(field, max_value, signed=False):
	r = field_to_channel(field, max_value, signed)
	b = 1.0 - r

	r = np.rint(255 * r**2).astype(np.uint8)
	g = np.zeros_like(r)
	b = np.rint(255 * b**2).astype(np.uint8)

	# Convert into single color array
	return np.transpose([r, g, b], (1, 2, 0))


def fields_to_rgb(fields, max_value, signed):
	assert len(fields) < 4 and len(fields) > 0

	if len(fields) == 1:
		return single_field_to_rgb(fields[0], max_value, signed)
		
	r = field_to_channel(fields[0], max_value, signed)
	g = field_to_channel(fields[1], max_value, signed)
		
	if len(fields) == 2:
		b = np.zeros_like(r)
	else:
		b = field_to_channel(fields[2], max_value, signed)

	return np.rint(255 * np.transpose([r, g, b], (1, 2, 0))).astype(np.uint8)


def plot_fields(fields, labels, colors, max_value, signed):
	fig = plt.figure()
	plt.ylim(-1 * max_value if signed else 0, max_value)
	x = np.arange(fields[0].size)
	plots = [plt.plot(x, f.reshape(-1), label=l)[0] for (f, l) in zip(fields, labels)]
	plt.legend(loc='upper right')

	return fig, plots

def plot_vector_fields(fields, labels, colors):
	fig = plt.figure()
	x, y = np.mgrid[0:fields[0].shape[0], 0:fields[0].shape[1]]
	plots = [plt.quiver(x, y, f[:,:,0], f[:,:,1], label=l, color=c, scale=10.) for (f, l, c) in zip(fields, labels, colors[0:len(fields)])]
	plt.legend(loc='upper right')

	return fig, plots

def render_fields(fields, labels, colors, max_value, signed):
	assert len(fields) == len(labels), "Number of fields and labels not matching up"
	if len(fields) > 3:
		fields = fields[:3]
		labels = labels[:3]
		print("more than three plots requested. Only showing first three")
	
	# TODO this still overrides the colors that are passed to the function
	colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
	fig = plt.figure()
	img = plt.imshow(fields_to_rgb(fields, max_value, signed))
	handles = [pch.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
	plt.legend(handles=handles, framealpha=0.5, loc='upper right')

	return fig, img


def combine_to_gif(image_dir, plot_name, ep_idx, ep_len, remove_frames):
	filenames = [os.path.join(image_dir, '%s%04d_%04d.png' % (plot_name, ep_idx, i)) for i in range(ep_len)]
	images = [imageio.imread(f) for f in filenames]
	output_file = os.path.join(image_dir, '%s%04d.gif' % (plot_name, ep_idx))
	imageio.mimsave(output_file, images)

	print('GIF written to %s' % output_file)

	if remove_frames:
		for filename in filenames:
			os.remove(filename)


class Viz:
	def __init__(self, field_dim):
		self.field_dim = field_dim

	def render(self, fields, labels, colors):
		if self.field_dim == 1:
			self._render_1d(fields, labels, colors)
		elif self.field_dim == 2:
			self._render_2d(fields, labels, colors)
		else:
			raise NotImplementedError()

	def _render_1d(self, fields, labels, colors):
		pass

	def _render_2d(self, fields, labels, colors):
		pass


class LiveViz(Viz):
	def __init__(self, field_dim, max_value, signed):
		super().__init__(field_dim)
		self.fig = None
		self.img = None
		self.plots = None
		self.max_value = max_value
		self.signed = signed

	def _render_1d(self, fields, labels, colors):
		if self.fig is None:
			plt.ion()
			self.fig, self.plots = plot_fields(fields, labels, colors, self.max_value, self.signed)
		else:
			for i in range(len(self.plots)):
				self.plots[i].set_ydata(fields[i])

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

	def _render_2d(self, fields, labels, colors):
		if fields[0].shape[-1] == 1 or True:
			self._render_img(fields, labels, colors)
		elif fields[0].shape[-1] == 2:
			self._render_quiver(fields, labels, colors)
		else:
			raise NotImplementedError()

	def _render_quiver(self, fields, labels, colors):
		if self.fig is None:
			plt.ion()
			self.fig, self.plots = plot_vector_fields(fields, labels, colors)
		else:
			for i, p in enumerate(self.plots):
				p.set_UVC(fields[i][:,:,0], fields[i][:,:,1])

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()

	def _render_img(self, fields, labels, colors):
		if self.fig is None:
			plt.ion()
			self.fig, self.img = render_fields(fields, labels, colors, self.max_value, self.signed)
		else:
			self.img.set_data(fields_to_rgb(fields, self.max_value, self.signed))

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()


class GifViz(Viz):
	def __init__(self, category_name, plot_name, field_dim, epis_len, max_value, signed, remove_frames):
		super().__init__(field_dim)
		self.scene = phiflow.Scene.create(os.path.expanduser('~/phi/data/'), category_name, mkdir=True)
		self.image_dir = self.scene.subpath('images', create=True)
		self.data = []
		self.max_value = max_value
		self.signed = signed
		self.plot_name = plot_name
		self.epis_len = epis_len
		self.remove_frames = remove_frames
		self.epis_idx = 0
		self.step_idx = 0

	def _render_1d(self, fields, labels, colors):
		plot_fields(fields, labels, colors, self.max_value, self.signed)

		path = self._get_path()
		plt.savefig(path)
		plt.close()

		if path:
			print('Frame written to %s' % path)

		self.step_idx += 1

		if self.step_idx == self.epis_len:
			combine_to_gif(self.image_dir, self.plot_name, self.epis_idx, self.epis_len, self.remove_frames)
			self.step_idx = 0
			self.epis_idx += 1

	def _render_2d(self, fields, labels, colors):
		fig, _ = render_fields(fields, labels, colors, self.max_value, self.signed)
		path = self._get_path()
		plt.savefig(path)
		plt.close()

		if path:
			print('Frame written to %s' % path)
		
		self.step_idx += 1

		if self.step_idx == self.epis_len:
			combine_to_gif(self.image_dir, self.plot_name, self.epis_idx, self.epis_len, self.remove_frames)
			self.step_idx = 0
			self.epis_idx += 1

	def _get_path(self):
		return os.path.join(self.image_dir, '%s%04d_%04d.png' % (self.plot_name, self.epis_idx, self.step_idx))


class PngViz(Viz):
	def __init__(self, category_name, plot_name, field_dim, epis_len, max_value, signed):
		super().__init__(field_dim)
		self.scene = phiflow.Scene.create(os.path.expanduser('~/phi/data/'), category_name, mkdir=True)
		self.image_dir = self.scene.subpath('images', create=True)
		self.data = []
		self.max_value = max_value
		self.signed = signed
		self.plot_name = plot_name
		self.epis_len = epis_len
		self.step_idx = 0
		self.epis_idx = 0

	def _render_1d(self, fields, labels, _colors):
		fields = fields[0:3]
		labels = labels[0:3]
		self.data.append(fields)

		self.step_idx += 1

		if self.step_idx == self.epis_len:
			for i in range(len(labels)):
				fig = plt.figure()
				plt.ylim(-1 * self.max_value if self.signed else 0, self.max_value)
				x = np.arange(self.data[0][0].size)
				colors = ['#%x%x%x%xff' % (c // 16, c % 16, 15 - (c // 16), 15 - (c % 16)) for c in [(256 * v) // (self.epis_len) for v in range(self.epis_len)]]
				plots = [plt.plot(x, f, c=c)[0] for (f, c) in zip([fields[i] for fields in self.data], colors)]

				path = os.path.join(self.image_dir, '%s%04d_%s.png' % (self.plot_name, self.epis_idx, labels[i]))
				plt.savefig(path)
				plt.close()
			self.data = []
			self.step_idx = 0
			self.epis_idx += 1

	def _render_2d(self, fields, labels, colors):
		raise NotImplementedError()