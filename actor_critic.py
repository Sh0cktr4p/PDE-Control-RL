from spinup.algos.pytorch.ppo import core
import torch
import numpy as np
import traceback


class RNN(torch.nn.Module):

	def __init__(self, input_shape, output_dim, sizes, activation, output_activation=torch.nn.Identity):
		super().__init__()
		print('Using Recurrent Network')
		self.x_size = np.prod(input_shape)
		self.h_size = sizes[0]
		self.y_size = output_dim
		self.flt = torch.nn.Flatten()
		self.rnn = torch.nn.GRU(self.x_size, self.h_size, batch_first=True)
		layers = []

		for j in range(len(sizes)-1):
			act = activation if j < len(sizes)-2 else output_activation
			layers += [torch.nn.Linear(sizes[j], sizes[j+1]), act()]

		self.seq = torch.nn.Sequential(*layers)
		self.h = self.init_hidden()
		self.hid_buf = []
		self.first_step = True

	def forward(self, x):
		x = x.view(-1, 1, self.x_size).float()
		if x.shape[0] == 1:
			if self.first_step:
				self.first_step = False
				self.hid_buf = []
			self.hid_buf.append(self.h)
			y, self.h = self.rnn(x, self.h)
			self.h.detach_()
		else:
			self.first_step = True
			y, _ = self.rnn(x, torch.cat(self.hid_buf[:-1], 1))
			self.h = self.init_hidden()
		y = self.seq(y)
		y = y.view(-1, self.y_size)
		return y

	def init_hidden(self):
		return torch.zeros((1, 1, self.h_size), requires_grad=True)


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


class UNET(torch.nn.Module):
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

		if len(input_shape) == 3:
			conv = torch.nn.Conv2d
			maxp = torch.nn.MaxPool2d
			cvtp = torch.nn.ConvTranspose2d

		print(output_channels)

		base_filter_count = 8

		self.blocks = torch.nn.ModuleList()

		self.blocks.append(torch.nn.Sequential(
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

		if len(x.shape) == 3:
			x = x.permute(0, 2, 1)
		elif len(x.shape) == 4:
			x = x.permute(0, 3, 1, 2)

		for i in range(len(self.blocks) // 2):
			x = self.blocks[i](x)
			block_outputs.append(x)

		x = self.blocks[len(self.blocks) // 2](x)

		for i in range(len(self.blocks) // 2):
			block_idx = len(self.blocks) // 2 + i + 1
			x = torch.cat((x, block_outputs[-(i+1)]), 1)
			x = self.blocks[block_idx](x)

		return x


class FCN(torch.nn.Module):

	def __init__(self, input_shape, output_dim, sizes, activation, output_activation=torch.nn.Identity):
		super().__init__()
		print('Using fully connected network')
		layers = [torch.nn.Flatten()]
		ext_sizes = [np.prod(input_shape)] + list(sizes) + [output_dim]

		for i in range(len(sizes)):
			act = activation if i < len(sizes) - 1 else output_activation
			layers += [torch.nn.Linear(ext_sizes[i], ext_sizes[i+1]), act()]

		self.seq = torch.nn.Sequential(*layers)

	def forward(self, x):
		return self.seq(x)


class MLPCategoricalActor(core.Actor):

	def __init__(self, obs_shape, act_dim, hidden_sizes, activation, network):
		super().__init__()
		self.logits_net = network(obs_shape, act_dim, list(hidden_sizes), activation).to('cuda')

	def _distribution(self, obs):
		logits = self.logits_net(obs.to('cuda')).to('cpu')
		return core.Categorical(logits=logits)

	def _log_prob_from_distribution(self, pi, act):
		return pi.log_prob(act)


class MLPGaussianActor(core.Actor):

	def __init__(self, obs_shape, act_dim, hidden_sizes, activation, network):
		super().__init__()
		log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
		self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
		self.mu_net = network(obs_shape, act_dim, list(hidden_sizes), activation).to('cuda')

	def _distribution(self, obs):
		mu = self.mu_net(obs.to('cuda')).to('cpu')
		std = torch.exp(self.log_std)
		return core.Normal(mu, std)

	def _log_prob_from_distribution(self, pi, act):
		return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(torch.nn.Module):

	def __init__(self, obs_shape, hidden_sizes, activation, network):
		super().__init__()
		self.v_net = network(obs_shape, 1, list(hidden_sizes), activation).to('cuda')

	def forward(self, obs):
		return torch.squeeze(self.v_net(obs.to('cuda')).to('cpu'), -1) # Critical to ensure v has right shape.



class MLPActorCritic(torch.nn.Module):

	def __init__(self, observation_space, action_space, 
				 hidden_sizes=(64,64), activation=torch.nn.Tanh, 
				 network=FCN):
		super().__init__()

		obs_shape = observation_space.shape

		# policy builder depends on action space
		if isinstance(action_space, core.Box):
			self.pi = MLPGaussianActor(obs_shape, action_space.shape[0], hidden_sizes, activation, network)
		elif isinstance(action_space, core.Discrete):
			self.pi = MLPCategoricalActor(obs_shape, action_space.n, hidden_sizes, activation, network)

		# build value function
		self.v  = MLPCritic(obs_shape, hidden_sizes, activation, network)

	def step(self, obs):
		obs = torch.as_tensor(np.expand_dims(obs.numpy(), 0))
		with torch.no_grad():
			pi = self.pi._distribution(obs)
			a = pi.sample()
			logp_a = self.pi._log_prob_from_distribution(pi, a)
			v = self.v(obs)
		return a.numpy(), v.numpy(), logp_a.numpy()

	def act(self, obs):
		return self.step(obs)[0]
