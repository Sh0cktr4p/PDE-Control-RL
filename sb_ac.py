from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import gym
from typing import Any, Callable, Dict, Optional, Tuple, Type
import numpy as np

class IdFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space):
        super().__init__(observation_space, np.prod(observation_space.shape))
        self._observation_space = observation_space
        self._features_dim = np.prod(observation_space.shape)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations

class CustomNetwork(torch.nn.Module):
    def __init__(
        self,
        pi_net: Type[torch.nn.Module],
        vf_net: Type[torch.nn.Module],
        observation_space: gym.Space,
        activation: Type[torch.nn.Module],
        latent_dim_pi: int,
        latent_dim_vf: int,
        pi_kwargs: Optional[Dict[str, Any]],
        vf_kwargs: Optional[Dict[str, Any]],
    ):
        super(CustomNetwork, self).__init__()

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf

        obs_shape = observation_space.shape

        # Policy network
        self.policy_net = pi_net(
            input_shape=obs_shape,
            output_dim=latent_dim_pi,
            activation=activation,
            **pi_kwargs
        )

        # Value network
        self.value_net = vf_net(
            input_shape=obs_shape,
            output_dim=latent_dim_vf,
            activation=activation,
            **vf_kwargs
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy_net(features), self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable[[float], float],
        pi_net: Type[torch.nn.Module],
        vf_net: Type[torch.nn.Module],
        vf_latent_dim: int,
        pi_kwargs: Optional[Dict[str, Any]] = None,
        vf_kwargs: Optional[Dict[str, Any]] = None,
        activation_fn: Type[torch.nn.Module] = torch.nn.ReLU,
        *args,
        **kwargs,
    ):
        self.pi_net = pi_net
        self.vf_net = vf_net

        self.pi_kwargs = pi_kwargs
        self.vf_kwargs = vf_kwargs

        self.pi_latent_dim = np.prod(action_space.shape)
        self.vf_latent_dim = vf_latent_dim

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            None,
            activation_fn,
            features_extractor_class=IdFeatureExtractor,
            *args,
            **kwargs,
        )

        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(
            self.pi_net, 
            self.vf_net, 
            self.observation_space, 
            self.activation_fn, 
            self.pi_latent_dim, 
            self.vf_latent_dim, 
            self.pi_kwargs, 
            self.vf_kwargs
        )
