from spinup.algos.pytorch.ppo import core
import torch
import numpy as np


def mlp(obs_shape, sizes, activation, output_activation=torch.nn.Identity):
	layers = []

	print(obs_shape)

	if(tuple(obs_shape[1:]) == (16,16)):
		layers += [torch.nn.Conv2d(obs_shape[0], 16, 3, padding=1), torch.nn.MaxPool2d(2)]
		layers += [torch.nn.Conv2d(16, 8, 3, padding=1), torch.nn.MaxPool2d(2)]
		layers += [torch.nn.Conv2d(8, 8, 3, padding=1), torch.nn.MaxPool2d(2)]
		layers += [torch.nn.Flatten()]
		sizes[0] = 32
	else:
		layers += [torch.nn.Flatten(), torch.nn.Linear(np.prod(obs_shape), sizes[0]), activation()]
	
	for j in range(len(sizes)-1):
		act = activation if j < len(sizes)-2 else output_activation
		layers += [torch.nn.Linear(sizes[j], sizes[j+1]), act()]
	return torch.nn.Sequential(*layers)


class MLPCategoricalActor(core.Actor):

	def __init__(self, obs_shape, act_dim, hidden_sizes, activation):
		super().__init__()
		self.logits_net = mlp(obs_shape, list(hidden_sizes) + [act_dim], activation)

	def _distribution(self, obs):
		logits = self.logits_net(obs)
		return core.Categorical(logits=logits)

	def _log_prob_from_distribution(self, pi, act):
		return pi.log_prob(act)


class MLPGaussianActor(core.Actor):

	def __init__(self, obs_shape, act_dim, hidden_sizes, activation):
		super().__init__()
		log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
		self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
		self.mu_net = mlp(obs_shape, list(hidden_sizes) + [act_dim], activation)

	def _distribution(self, obs):
		mu = self.mu_net(obs)
		std = torch.exp(self.log_std)
		return core.Normal(mu, std)

	def _log_prob_from_distribution(self, pi, act):
		return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(torch.nn.Module):

	def __init__(self, obs_shape, hidden_sizes, activation):
		super().__init__()
		self.v_net = mlp(obs_shape, list(hidden_sizes) + [1], activation)

	def forward(self, obs):
		return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(torch.nn.Module):


	def __init__(self, observation_space, action_space, 
				 hidden_sizes=(64,64), activation=torch.nn.Tanh):
		super().__init__()

		obs_shape = observation_space.shape

		# policy builder depends on action space
		if isinstance(action_space, core.Box):
			self.pi = MLPGaussianActor(obs_shape, action_space.shape[0], hidden_sizes, activation)
		elif isinstance(action_space, core.Discrete):
			self.pi = MLPCategoricalActor(obs_shape, action_space.n, hidden_sizes, activation)

		# build value function
		self.v  = MLPCritic(obs_shape, hidden_sizes, activation)

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