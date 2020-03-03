import time
import sys
import os
import numpy as np
import pyglet
import phi.flow
import matplotlib.pyplot as plt
import imageio

def field_to_rgb(field, max_value):
		obs = field
		
		assert obs.shape[-1] < 3, "3D visualization not (yet) supported"

		height, width = field.shape[-3:-1]

		# Visualization should display field vector length
		obs = np.linalg.norm(obs, axis=-1).reshape(height, width)

		# Get the color values
		r = np.clip((obs + max_value) / (2.0 * max_value), 0.0, 1.0)
		b = 1.0 - r

		r = np.rint(255 * r**2).astype(np.uint8)
		g = np.zeros_like(r)
		b = np.rint(255 * b**2).astype(np.uint8)

		# Convert into single color array
		return np.transpose([r, g, b], (1, 2, 0))


def plot_fields(fields, labels):
	fig = plt.figure()
	x = np.arange(fields[0].size)
	plots = [plt.plot(x, f, label=l)[0] for (f, l) in zip(fields, labels)]
	plt.legend(loc='upper right')

	return fig, plots

class Renderer:
	def __init__(self, display=None):
		self.window = None
		self.isopen = False
		self.display = display
		self.width = 0
		self.height = 0

	def render(self, field, frame_rate, max_value, width, height):
		tic = time.time()

		rgb = field_to_rgb(field, max_value)

		if self.window is None:
			self.window = pyglet.window.Window(width=width, height=height, 
				display=self.display, vsync=False, resizable=True)

			self.width = width
			self.height = height
			self.isopen = True

			@self.window.event
			def on_resize(width, height):
				self.width = width
				self.height = height

			@self.window.event
			def on_close():
				self.isopen = False

		assert len(rgb.shape) == 3
		image = pyglet.image.ImageData(rgb.shape[1], rgb.shape[0],
			'RGB', rgb.tobytes(), pitch=rgb.shape[1]*-3)

		pyglet.gl.glTexParameteri(pyglet.gl.GL_TEXTURE_2D,
			pyglet.gl.GL_TEXTURE_MAG_FILTER, pyglet.gl.GL_NEAREST)
		texture = image.get_texture()
		texture.width = self.width
		texture.height = self.height
		self.window.clear()
		self.window.switch_to()
		self.window.dispatch_events()
		texture.blit(0, 0)
		self.window.flip()
		toc = time.time()
		sleep_time = 1 / frame_rate - (toc - tic)
		if sleep_time > 0:
			time.sleep(sleep_time)

	def close(self):
		if self.isopen and sys.meta_path:
			self.window.close()
			self.isopen = False

	def __del__(self):
		self.close()
		

class FilePlotter:
	def __init__(self, category_name):
		self.scene = phi.flow.Scene.create(os.path.expanduser('~/phi/data/'), category_name, mkdir=True)
		self.image_dir = self.scene.subpath('images', create=True)

	def render(self, fields, labels, plot_name, ep_idx, step_idx, ep_len):
		fig, _ = plot_fields(fields, labels)

		path = os.path.join(self.image_dir, '%s%04d_%04d.png' % (plot_name, ep_idx, step_idx))
		plt.savefig(path)
		plt.close()

		if path:
			print('Frame written to %s' % path)

		if step_idx == ep_len - 1:
			filenames = [os.path.join(self.image_dir, '%s%04d_%04d.png' % (plot_name, ep_idx, i)) for i in range(ep_len)]
			images = [imageio.imread(f) for f in filenames]
			output_file = os.path.join(self.image_dir, '%s%04d.gif' % (plot_name, ep_idx))
			imageio.mimsave(output_file, images)

			print('GIF written to %s' % output_file)

			for filename in filenames:
				os.remove(filename)


class LivePlotter:
	def __init__(self):
		self.fig = None
		self.plots = None

	def render(self, fields, labels):
		if self.fig is None:
			plt.ion()
			self.fig, self.plots = plot_fields(fields, labels)
		else:
			for i in range(len(self.plots)):
				self.plots[i].set_ydata(fields[i])

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()