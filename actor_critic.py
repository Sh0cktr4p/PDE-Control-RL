from spinup.algos.pytorch.ppo import core
import torch
import numpy as np
import traceback


class RNN(torch.nn.Module):

	def __init__(self, obs_shape, sizes, activation, output_activation=torch.nn.Identity):
		super().__init__()
		print('Using Recurrent Network')
		self.x_size = np.prod(obs_shape)
		self.h_size = sizes[0]
		self.y_size = sizes[-1]
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


class UNT(torch.nn.Module):

	def __init__(self, obs_shape, sizes, activation, output_activation=torch.nn.Identity):
		super().__init__()

		fcs = [8, 16, 32]
		ics = 4
		ocs = 2

		ks = 3
		st = 1
		pd = 1

		print(obs_shape)
		print(sizes)

		self.blocks = torch.nn.ModuleList()

		# [C, C]
		self.blocks.append(torch.nn.Sequential(*[
			torch.nn.Conv2d(ics, fcs[0], ks, st, pd), activation(),
			torch.nn.Conv2d(fcs[0], fcs[0], ks, st, pd), activation()]))

		# [P, C, C]
		for i in range(len(fcs) - 2):
			self.blocks.append(torch.nn.Sequential(*[
				torch.nn.MaxPool2d(2),
				torch.nn.Conv2d(fcs[i], fcs[i+1], ks, st, pd), activation(),
				torch.nn.Conv2d(fcs[i+1], fcs[i+1], ks, st, pd), activation()]))
		
		# [P, C, C, U]
		self.blocks.append(torch.nn.Sequential(*[
			torch.nn.MaxPool2d(2),
			torch.nn.Conv2d(fcs[-2], fcs[-1], ks, st, pd), activation(),
			torch.nn.Conv2d(fcs[-1], fcs[-1], ks, st, pd), activation(),
			torch.nn.ConvTranspose2d(fcs[-1], fcs[-2], 2, 2)]))

		# [C, C, U]
		for i in range(len(fcs) - 2):
			self.blocks.append(torch.nn.Sequential(*[
				torch.nn.Conv2d(fcs[-(i+1)], fcs[-(i+2)], ks, st, pd), activation(),
				torch.nn.Conv2d(fcs[-(i+2)], fcs[-(i+2)], ks, st, pd), activation(),
				torch.nn.ConvTranspose2d(fcs[-(i+2)], fcs[-(i+3)], 2, 2)]))

		# [C, C, C]
		mods = [
			torch.nn.Conv2d(fcs[1], fcs[0], ks, st, pd), activation(),
			torch.nn.Conv2d(fcs[0], fcs[0], ks, st, pd), activation(),
			torch.nn.Conv2d(fcs[0], ocs, ks, st, pd), 
			torch.nn.Flatten(), output_activation()]
		
		if sizes[-1] == 1:
			print('Additional layer for outputting only one value')
			mods.append(torch.nn.Linear(obs_shape[0] * obs_shape[1] * ocs, 1))

		self.blocks.append(torch.nn.Sequential(*mods))

		print('Using fully convolutional U-Net model')
		print('Number of blocks: %d' % len(self.blocks))

	def forward(self, x):
		block_outputs = []

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

	def __init__(self, obs_shape, sizes, activation, output_activation=torch.nn.Identity):
		super().__init__()
		print('Using fully connected network')
		layers = [torch.nn.Flatten()]
		ext_sizes = [np.prod(obs_shape)] + list(sizes)

		for i in range(len(sizes)):
			act = activation if i < len(sizes) - 1 else output_activation
			layers += [torch.nn.Linear(ext_sizes[i], ext_sizes[i+1]), act()]

		self.seq = torch.nn.Sequential(*layers)

	def forward(self, x):
		return self.seq(x)

 
def mlp(obs_shape, sizes, activation, output_activation=torch.nn.Identity):
	#return RNN(obs_shape, sizes, activation, output_activation)
	
	layers = []

	print(obs_shape)

	obs_shape = tuple([obs_shape[-1]] + list(obs_shape[:-1]))

	if(len(obs_shape) == 3) and np.prod(obs_shape[1:]) % 64 == 0 and obs_shape[1] == obs_shape[2]:
		print('Using convolutional layers')
		layers += [torch.nn.Conv2d(obs_shape[0], 8, 3, padding=1), torch.nn.MaxPool2d(2), activation()]
		layers += [torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.MaxPool2d(2), activation()]
		layers += [torch.nn.Conv2d(16, 32, 3, padding=1), activation()]
		layers += [torch.nn.Flatten()]

		sizes[0] = np.prod(obs_shape[1:]) // 2
	else:
		print('Using fully connected model')
		layers += [torch.nn.Flatten(), torch.nn.Linear(np.prod(obs_shape), sizes[0]), activation()]
	
	for j in range(len(sizes)-1):
		act = activation if j < len(sizes)-2 else output_activation
		layers += [torch.nn.Linear(sizes[j], sizes[j+1]), act()]
	return torch.nn.Sequential(*layers)


class MLPCategoricalActor(core.Actor):

	def __init__(self, obs_shape, act_dim, hidden_sizes, activation, network):
		super().__init__()
		self.logits_net = network(obs_shape, list(hidden_sizes) + [act_dim], activation)

	def _distribution(self, obs):
		logits = self.logits_net(obs)
		return core.Categorical(logits=logits)

	def _log_prob_from_distribution(self, pi, act):
		return pi.log_prob(act)


class MLPGaussianActor(core.Actor):

	def __init__(self, obs_shape, act_dim, hidden_sizes, activation, network):
		super().__init__()
		log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
		self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
		self.mu_net = network(obs_shape, list(hidden_sizes) + [act_dim], activation)

	def _distribution(self, obs):
		mu = self.mu_net(obs)
		std = torch.exp(self.log_std)
		return core.Normal(mu, std)

	def _log_prob_from_distribution(self, pi, act):
		return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(torch.nn.Module):

	def __init__(self, obs_shape, hidden_sizes, activation, network):
		super().__init__()
		self.v_net = network(obs_shape, list(hidden_sizes) + [1], activation)

	def forward(self, obs):
		return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



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
