import torch
import numpy as np
import traceback
from typing import List


class FCN(torch.nn.Module):

	def __init__(self, input_shape, output_dim, sizes, activation, output_activation=torch.nn.Identity):
		super().__init__()
		print('Using fully connected network')
		layers = [torch.nn.Flatten()]
		ext_sizes = [np.prod(input_shape)] + list(sizes) + [output_dim]
		print(ext_sizes)

		for i in range(len(ext_sizes) - 1):
			act = activation if i < len(sizes) - 1 else output_activation
			layers += [torch.nn.Linear(ext_sizes[i], ext_sizes[i+1]), act()]

		self.seq = torch.nn.Sequential(*layers)

	def forward(self, x):
		return self.seq(x)


class CNN(torch.nn.Module):

	def __init__(self, obs_shape, sizes, activation, output_activation=torch.nn.Identity):
		super().__init__()
		print('Using Convolutional Network')

		conv_sizes = sizes[0]
		lin_sizes = sizes[1] + [sizes[2]]

		layers = []

		# Add input channels to conv sizes
		ext_conv_sizes = [obs_shape[-1]] + list(conv_sizes)
		print('Filter counts: ', ext_conv_sizes)

		for i in range(len(conv_sizes)):
			layers += [torch.nn.Conv2d(ext_conv_sizes[i], ext_conv_sizes[i+1], 3, padding=1), torch.nn.MaxPool2d(2), activation()]

		layers.append(torch.nn.Flatten())

		# Add flatten layer output to lin sizes
		ext_lin_sizes = [(np.prod(obs_shape[:-1]) * conv_sizes[-1]) // 2 ** (2 * len(conv_sizes))] + list(lin_sizes)
		print('Fully connected layer sizes: ', ext_lin_sizes)

		for j in range(len(lin_sizes)):
			act = activation if j < len(lin_sizes)-1 else output_activation
			layers += [torch.nn.Linear(ext_lin_sizes[j], ext_lin_sizes[j+1]), act()]

		self.seq = torch.nn.Sequential(*layers)

	def forward(self, x):
		return self.seq(x.permute(0, 3, 1, 2))


class CNN_FUNNEL(torch.nn.Module):
	def __init__(self, input_shape, output_dim, sizes, activation, output_activation=torch.nn.Identity):
		super().__init__()

		input_dim = len(input_shape) - 1

		if input_dim == 1:
			conv = torch.nn.Conv1d
			pool = torch.nn.MaxPool1d
			pad = OneSidedPadding1d
			perm = Permute([0, 2, 1])
		elif input_dim == 2:
			assert input_shape[0] == input_shape[1]
			conv = torch.nn.Conv2d
			pool = torch.nn.MaxPool2d
			pad = OneSidedPadding2d
			perm = Permute([0, 3, 1, 2])

		input_width = input_shape[0]

		num_levels = 0

		while 2 ** num_levels < input_width:
			num_levels += 1

		assert len(sizes) == num_levels

		filter_counts = [input_shape[-1]] + sizes

		pad_width = 2 ** num_levels - input_width
		layers = [
			perm,
			pad(amount=pad_width, mode='constant'),
		]

		for i in range(num_levels):
			layers += [
				conv(filter_counts[i], filter_counts[i+1], 3, 1, 1),
				activation(),
				pool(2),
			]

		layers += [
			conv(filter_counts[-1], output_dim, 1, 1, 0),
			output_activation(),
			torch.nn.Flatten(),
		]

		self.net = torch.nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x)


class RES_FUNNEL(torch.nn.Module):
	def __init__(self, input_shape, output_dim, sizes, activation, output_activation=torch.nn.Identity):
		super().__init__()

		input_dim = len(input_shape) - 1

		if input_dim == 1:
			res_block = ResBlock1d
			conv = torch.nn.Conv1d
			pad = OneSidedPadding1d
			perm = Permute([0, 2, 1])
		elif input_dim == 2:
			assert input_shape[0] == input_shape[1]
			res_block = ResBlock2d
			conv = torch.nn.Conv2d
			pad = OneSidedPadding2d
			perm = Permute([0, 3, 1, 2])

		input_width = input_shape[0]

		num_levels = 0

		while 2 ** num_levels < input_width:
			num_levels += 1

		assert len(sizes) == num_levels

		filter_counts = sizes + [output_dim]

		pad_width = 2 ** num_levels - input_width
		layers = [
			perm,
			pad(amount=pad_width, mode='constant'),
		]

		for i in range(num_levels):
			layers += [
				conv(filter_counts[i], filter_counts[i+1], 2, 2, 0),
				activation(),
				res_block(filter_counts[i+1], 3, 1, 1, activation),
				res_block(filter_counts[i+1], 3, 1, 1, activation),
			]

		layers += [
			output_activation(),
			torch.nn.Flatten(),
		]

		self.net = torch.nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x)


class SIMP_UNET(torch.nn.Module):
	def __init__(self, input_shape, output_dim, sizes, activation, output_activation=torch.nn.Identity):
		super().__init__()

		num_levels = len(sizes)

		assert num_levels > 1

		print('Input shape: %s' % str(input_shape))
		print('Output dim: %s' % str(output_dim))

		input_channels = input_shape[-1]

		# Number of output channels is derived from the size of the field and the overall action parameters
		# Would be zero in the case of the value net => clamp to one
		output_channels = max(1, output_dim // np.prod(input_shape[:-1]))

		kernel_size = 3
		stride = 1
		padding = 1

		maxpool_size = 2

		transpose_conv_kernel_size = 2
		transpose_conv_stride = 2

		conv = torch.nn.Conv1d
		maxp = torch.nn.MaxPool1d
		cvtp = torch.nn.ConvTranspose1d
		perm_fwd = Permute([0, 2, 1])
		perm_bwd = Permute([0, 2, 1])

		if len(input_shape) == 3:
			print("Using 2D modules!")
			conv = torch.nn.Conv2d
			maxp = torch.nn.MaxPool2d
			cvtp = torch.nn.ConvTranspose2d
			perm_fwd = Permute([0, 3, 1, 2])
			perm_bwd = Permute([0, 2, 3, 1])

		print(output_channels)

		base_filter_count = 8

		self.blocks = torch.nn.ModuleList()

		self.blocks.append(torch.nn.Sequential(
			perm_fwd,
			conv(input_channels, base_filter_count, kernel_size, stride, padding), activation(),
			conv(base_filter_count, base_filter_count, kernel_size, stride, padding), activation()
		))

		for i in range(num_levels - 1):
			self.blocks.append(torch.nn.Sequential(
				maxp(maxpool_size),
				conv(base_filter_count * 2**i, base_filter_count * 2**(i+1), kernel_size, stride, padding), activation(),
				conv(base_filter_count * 2**(i+1), base_filter_count * 2**(i+1), kernel_size, stride, padding), activation()
			))

		self.blocks.append(torch.nn.Sequential(
			maxp(maxpool_size),
			conv(base_filter_count * 2**(num_levels-1), base_filter_count * 2**num_levels, kernel_size, stride, padding), activation(),
			conv(base_filter_count * 2**num_levels, base_filter_count * 2**num_levels, kernel_size, stride, padding), activation(),
			cvtp(base_filter_count * 2**num_levels, base_filter_count * 2**(num_levels-1), transpose_conv_kernel_size, transpose_conv_stride)
		))

		for i in range(num_levels - 1):
			self.blocks.append(torch.nn.Sequential(
				conv(base_filter_count * 2**(num_levels-i), base_filter_count * 2**(num_levels-i-1), kernel_size, stride, padding), activation(),
				conv(base_filter_count * 2**(num_levels-i-1), base_filter_count * 2**(num_levels-i-1), kernel_size, stride, padding), activation(),
				cvtp(base_filter_count * 2**(num_levels-i-1), base_filter_count * 2**(num_levels-i-2), transpose_conv_kernel_size, transpose_conv_stride),
			))

		mods = [
			conv(base_filter_count * 2, base_filter_count, kernel_size, stride, padding), activation(),
			conv(base_filter_count, base_filter_count, kernel_size, stride, padding), activation(),
			conv(base_filter_count, output_channels, stride, padding), output_activation(),
			perm_bwd,
			torch.nn.Flatten()
		]

		if output_dim == 1:
			mods += [torch.nn.Linear(np.prod(input_shape[:-1]), output_dim)]

		self.blocks.append(torch.nn.Sequential(*mods))

		print([sum(p.numel() for p in block.parameters()) for block in self.blocks])

		print('Using U-Net with %d levels' % num_levels)
		print('Number of Blocks: %d' % len(self.blocks))

	def forward(self, x):
		block_outputs = []

		for i in range(len(self.blocks) // 2):
			x = self.blocks[i](x)
			block_outputs.append(x)

		x = self.blocks[len(self.blocks) // 2](x)

		for i in range(len(self.blocks) // 2):
			block_idx = len(self.blocks) // 2 + i + 1
			x = torch.cat((x, block_outputs[-(i+1)]), 1)
			x = self.blocks[block_idx](x)

		return x


class RES_UNET(torch.nn.Module):
	def __init__(self, input_shape, output_dim, sizes, activation, output_activation=torch.nn.Identity):
		super().__init__()

		if type(input_shape) != tuple:
			input_shape = input_shape.shape

		#print("Input shape: %s" % str(input_shape))

		# input_shape: (w, c) or (w, h, c)
		input_dim = len(input_shape) - 1

		if input_dim == 1:
			conv = torch.nn.Conv1d
			pad = OneSidedPadding1d
			res_block = ResBlock1d
			perm_fwd = Permute([0, 2, 1])
			perm_bwd = Permute([0, 2, 1])
			upsample_mode = 'linear'
		elif input_dim == 2:
			conv = torch.nn.Conv2d
			pad = OneSidedPadding2d
			res_block = ResBlock2d
			perm_fwd = Permute([0, 3, 1, 2])
			perm_bwd = Permute([0, 2, 3, 1])
			upsample_mode = 'bilinear'
		else:
			raise NotImplementedError()

		self.num_levels = len(sizes)

		input_channels = input_shape[-1]
		pad_width = sum([2**i for i in range(self.num_levels)])

		output_channels = max(1, output_dim // np.prod(input_shape[:-1]))   

		filter_counts = [input_channels] + sizes

		self.encoder_blocks = torch.nn.ModuleList()
		self.decoder_blocks = torch.nn.ModuleList()

		# Putting channels before spacial dimensions + initial padding
		self.preprocess = torch.nn.Sequential(
			perm_fwd,
			pad(amount=pad_width),
		)

		# Encoder
		for i in range(self.num_levels):
			self.encoder_blocks.append(torch.nn.Sequential(
				conv(filter_counts[i], filter_counts[i+1], 2, 2, 0),
				activation(),
				res_block(filter_counts[i+1], 3, 1, 1, activation),
				res_block(filter_counts[i+1], 3, 1, 1, activation),
			))

		# Bottleneck
		self.bottleneck = torch.nn.Sequential(
			res_block(filter_counts[self.num_levels], 3, 1, 1, activation),
			res_block(filter_counts[self.num_levels], 3, 1, 1, activation),
			res_block(filter_counts[self.num_levels], 3, 1, 1, activation),
			torch.nn.Upsample(scale_factor=2, mode=upsample_mode),
		)

		# Decoder
		for i in range(self.num_levels)[::-1]:
			if i > 0:
				self.decoder_blocks.append(torch.nn.Sequential(
					pad(1, 'constant'),
					conv(filter_counts[i] + 16, 16, 2, 1, 0),
					activation(),
					res_block(16, 3, 1, 1, activation),
					res_block(16, 3, 1, 1, activation),
					torch.nn.Upsample(scale_factor=2, mode=upsample_mode),
				))
			else:
				# Last decoder step
				self.decoder_blocks.append(torch.nn.Sequential(
					pad(1, 'constant'),
					conv(filter_counts[0] + 16, output_channels, 2, 1, 0),
					output_activation(),
					perm_bwd,
					torch.nn.Flatten(),
				))

		shave_amts = [2**(i+1) - 1 for i in range(self.num_levels)]

		self.shavers = torch.nn.ModuleList([pad(-i, 'constant') for i in shave_amts])

		#print('Using U-Net with %d levels' % self.num_levels)

	def forward(self, x):
		y = self.preprocess(x)
		res = [y]

		for i in range(self.num_levels):
			y = self.encoder_blocks[i](y)
			res.insert(0, y)

		y = self.bottleneck(y)

		for i, res_data in enumerate(res[1:]):
			res_in = self.shavers[i](res_data)
			y = torch.cat((y, res_in), 1)
			y = self.decoder_blocks[i](y)

		return y


class ResBlock1d(torch.nn.Module):
	def __init__(self, fc, ks, st, pd, act):
		super().__init__()

		self.res_block = torch.nn.Sequential(
			torch.nn.Conv1d(fc, fc, ks, st, pd),
			act(),
			torch.nn.Conv1d(fc, fc, ks, st, pd)
		)

		self.final_act = act()

	def forward(self, x):
		return self.final_act(self.res_block(x) + x)


class ResBlock2d(torch.nn.Module):
	def __init__(self, fc, ks, st, pd, act):
		super().__init__()

		self.res_block = torch.nn.Sequential(
			torch.nn.Conv2d(fc, fc, ks, st, pd),
			act(),
			torch.nn.Conv2d(fc, fc, ks, st, pd)
		)

		self.final_act = act()

	def forward(self, x):
		return self.final_act(self.res_block(x) + x)


class Permute(torch.nn.Module):
	def __init__(self, dims):
		super().__init__()
		self.dims = dims

	def forward(self, x):
		return x.permute(*self.dims)


class OneSidedPadding1d(torch.nn.Module):
	def __init__(self, amount=1, mode='constant'):
		super().__init__()
		self.mode = mode
		self.amount = amount

	def forward(self, x):
		return torch.nn.functional.pad(x, (0, self.amount), mode=self.mode)


class OneSidedPadding2d(torch.nn.Module):
	def __init__(self,amount=1, mode='constant'):
		super().__init__()
		self.mode = mode
		self.amount = amount

	def forward(self, x):
		return torch.nn.functional.pad(x, (0, self.amount, 0, self.amount), mode=self.mode)
