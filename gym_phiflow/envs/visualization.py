import time
import sys
import os
import numpy as np
import pyglet
import phi.flow
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
	r = field_to_channel(max_value, field, signed)
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


def plot_fields(fields, labels, max_value, signed):
	fig = plt.figure()
	plt.ylim(-1 * max_value if signed else 0, max_value)
	x = np.arange(fields[0].size)
	plots = [plt.plot(x, f, label=l)[0] for (f, l) in zip(fields, labels)]
	plt.legend(loc='upper right')

	return fig, plots


def render_fields(fields, labels, max_value, signed):
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


class LiveViz:
	def __init__(self):
		pass

	def render(self, fields, labels, max_value, signed):
		pass


class FileViz:
	def __init__(self, category_name):
		pass

	def render(self, fields, labels, max_value, signed, plot_name, ep_idx, step_idx, ep_len, remove_frames):
		pass


class LiveRenderer(LiveViz):
	def __init__(self):
		self.fig = None
		self.img = None

	def render(self, fields, labels, max_value, signed):
		if self.fig is None:
			plt.ion()
			self.fig, self.img = render_fields(fields, labels, max_value, signed)
		else:
			self.img.set_data(fields_to_rgb(fields, max_value, signed))

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		

class FileRenderer(FileViz):
	def __init__(self, category_name):
		self.scene = phi.flow.Scene.create(os.path.expanduser('~/phi/data/'), category_name, mkdir=True)
		self.image_dir = self.scene.subpath('images', create=True)

	def render(self, fields, labels, max_value, signed, plot_name, ep_idx, step_idx, ep_len, remove_frames):
		if step_idx == ep_len:
			ep_idx -= 1

		fig, _ = render_fields(fields, labels, max_value, signed)

		path = os.path.join(self.image_dir, '%s%04d_%04d.png' % (plot_name, ep_idx, step_idx))
		plt.savefig(path)
		plt.close()

		if path:
			print('Frame written to %s' % path)
		
		if step_idx == ep_len:
			combine_to_gif(self.image_dir, plot_name, ep_idx, ep_len+1, remove_frames)


class LivePlotter(LiveViz):
	def __init__(self):
		self.fig = None
		self.plots = None

	def render(self, fields, labels, max_value, signed):
		if self.fig is None:
			plt.ion()
			self.fig, self.plots = plot_fields(fields, labels, max_value, signed)
		else:
			for i in range(len(self.plots)):
				self.plots[i].set_ydata(fields[i])

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()


class GifPlotter(FileViz):
	def __init__(self, category_name):
		self.scene = phi.flow.Scene.create(os.path.expanduser('~/phi/data/'), category_name, mkdir=True)
		self.image_dir = self.scene.subpath('images', create=True)
		self.data = []

	def render(self, fields, labels, max_value, signed, plot_name, ep_idx, step_idx, ep_len, remove_frames):
		#if step_idx == ep_len:
		#	ep_idx -= 1
		
		fig, _ = plot_fields(fields, labels, max_value, signed)

		path = os.path.join(self.image_dir, '%s%04d_%04d.png' % (plot_name, ep_idx, step_idx))
		plt.savefig(path)
		plt.close()

		if path:
			print('Frame written to %s' % path)

		if step_idx == ep_len - 1:
			combine_to_gif(self.image_dir, plot_name, ep_idx, ep_len, remove_frames)


class PngPlotter(FileViz):
	def __init__(self, category_name):
		self.scene = phi.flow.Scene.create(os.path.expanduser('~/phi/data/'), category_name, mkdir=True)
		self.image_dir = self.scene.subpath('images', create=True)
		self.data = []

	def render(self, fields, labels, max_value, signed, plot_name, ep_idx, step_idx, ep_len, remove_frames):
		fields = fields[0:3]
		labels = labels[0:3]
		self.data.append(fields)

		if step_idx == ep_len:
			for i in range(len(labels)):
				fig = plt.figure()
				plt.ylim(-1 * max_value if signed else 0, max_value)
				x = np.arange(self.data[0][0].size)
				colors = ['#%x%x%x%xff' % (c // 16, c % 16, 15 - (c // 16), 15 - (c % 16)) for c in [(256 * v) // (ep_len + 1) for v in range(ep_len + 1)]]
				plots = [plt.plot(x, f, c=c)[0] for (f, c) in zip([fields[i] for fields in self.data], colors)]

				path = os.path.join(self.image_dir, '%s%04d_%s.png' % (plot_name, ep_idx, labels[i]))
				plt.savefig(path)
				plt.close()
			self.data = []
