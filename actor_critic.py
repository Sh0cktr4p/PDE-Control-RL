from spinup.algos.pytorch.ppo import core
import networks
import torch
import numpy as np



class MLPCategoricalActor(core.Actor):

	def __init__(self, obs_shape, act_dim, hidden_sizes, activation, network, device):
		super().__init__()
		self.logits_net = network(obs_shape, act_dim, list(hidden_sizes), activation).to(device)
		self.device = device

	def _distribution(self, obs):
		logits = self.logits_net(obs.to(self.device)).to('cpu')
		return core.Categorical(logits=logits)

	def _log_prob_from_distribution(self, pi, act):
		return pi.log_prob(act)


class MLPGaussianActor(core.Actor):

	def __init__(self, obs_shape, act_dim, hidden_sizes, activation, network, device):
		super().__init__()
		log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
		self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
		self.mu_net = network(obs_shape, act_dim, list(hidden_sizes), activation).to(device)
		self.device = device

	def _distribution(self, obs):
		mu = self.mu_net(obs.to(self.device)).to('cpu')
		std = torch.exp(self.log_std)
		return core.Normal(mu, std)

	def _log_prob_from_distribution(self, pi, act):
		return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(torch.nn.Module):

	def __init__(self, obs_shape, hidden_sizes, activation, network, device):
		super().__init__()
		self.v_net = network(obs_shape, 1, list(hidden_sizes), activation).to(device)
		self.device = device

	def forward(self, obs):
		return torch.squeeze(self.v_net(obs.to(self.device)).to('cpu'), -1) # Critical to ensure v has right shape.



class MLPActorCritic(torch.nn.Module):

	def __init__(self, observation_space, action_space, 
				 pi_hidden_sizes=(64,64), vf_hidden_sizes=(64, 64), activation=torch.nn.Tanh, 
				 pi_network=networks.FCN, vf_network=networks.FCN, device='cpu'):
		super().__init__()

		obs_shape = observation_space.shape
		print("Observation space shape: %s" % str(obs_shape))
		# policy builder depends on action space
		if isinstance(action_space, core.Box):
			self.pi = MLPGaussianActor(obs_shape, action_space.shape[0], pi_hidden_sizes, activation, pi_network, device)
		elif isinstance(action_space, core.Discrete):
			self.pi = MLPCategoricalActor(obs_shape, action_space.n, pi_hidden_sizes, activation, pi_network, device)

		# build value function
		self.v  = MLPCritic(obs_shape, vf_hidden_sizes, activation, vf_network, device)

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
