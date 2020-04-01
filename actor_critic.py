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
		lin_sizes = sizes[1]

		layers = []

		# Add input channels to conv sizes
		ext_conv_sizes = obs_shape[0] + list(conv_sizes)

		for i in range(len(conv_sizes)):
			layers += [torch.nn.Conv2d(ext_conv_sizes[i], ext_conv_sizes[i+1], 3, padding=1), torch.nn.MaxPool2d(2), activation()]

		layers.append(torch.nn.Flatten())

		# Add flatten layer output to lin sizes
		ext_lin_sizes = [(np.prod(obs_shape) * conv_sizes[-1]) // 2 ** (2 * len(conv_sizes))] + list(lin_sizes)

		for j in range(len(lin_sizes)):
			act = activation if j < len(lin_sizes)-1 else output_activation
			layers += [torch.nn.Linear(ext_lin_sizes[j], ext_lin_sizes[j+1]), act()]

		self.seq = torch.nn.Sequential(*layers)

	def forward(self, x):
		return self.seq(x)


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
