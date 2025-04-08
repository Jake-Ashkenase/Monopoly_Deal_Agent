import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo.policies import ActorCriticPolicy
import gym
from typing import Dict, List, Tuple, Type, Union
import numpy as np

class MonopolyDealFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        # Custom preprocessing: concatenate all observations into a single vector
        # Calculate the total size of all observation spaces
        total_flat_size = (
            10 +  # Agent hand
            10 +  # Agent Board
            10 +  # Opponent Board
            6 +   # Agent Cash
            6 +   # Opponent Cash
            1 +   # Turn
            10    # Action mask
        )
        super().__init__(observation_space, features_dim=features_dim)
        
        self.linear = nn.Sequential(
            nn.Linear(total_flat_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Get batch size from first observation
        batch_size = observations["Agent hand"].shape[0] if len(observations["Agent hand"].shape) > 1 else 1

        # Convert and reshape all tensors to have same batch size
        agent_hand = observations["Agent hand"].float().reshape(batch_size, -1)[:, :10]
        agent_board = observations["Agent Board"].float().reshape(batch_size, -1)[:, :10]
        opponent_board = observations["Opponent Board"].float().reshape(batch_size, -1)[:, :10]
        agent_cash = observations["Agent Cash"].float().reshape(batch_size, -1)[:, :6]
        opponent_cash = observations["Opponent Cash"].float().reshape(batch_size, -1)[:, :6]
        turn = observations["Turn"].float().reshape(batch_size, -1)[:, :1]
        action_mask = observations["action_mask"].float().reshape(batch_size, -1)[:, :10]

        # Concatenate all features along the feature dimension (dim=1)
        combined = torch.cat([
            agent_hand,
            agent_board,
            opponent_board,
            agent_cash,
            opponent_cash,
            turn,
            action_mask
        ], dim=1)

        return self.linear(combined)


class CustomMaskedPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: callable,
        *args,
        **kwargs,
    ):
        # Initialize the base class with our custom feature extractor
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=MonopolyDealFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            *args,
            **kwargs,
        )

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions (ignored - always deterministic)
        :return: action, value and log probability of the action
        """
        # Extract features from the observations
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Get values from the value network
        values = self.value_net(latent_vf)

        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Get the action mask and convert to bool
        action_mask = obs["action_mask"].bool()
        
        # Get original logits
        logits = distribution.distribution.logits
        
        # Always use deterministic selection - mask invalid actions and take argmax
        invalid_action_mask = ~action_mask
        logits = logits.masked_fill(invalid_action_mask, float('-inf'))
        actions = torch.argmax(logits, dim=1)
        
        # Calculate log probabilities for the chosen actions
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions, entropy of the action distribution
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        # Apply action mask
        action_mask = obs["action_mask"].bool()
        logits = distribution.distribution.logits
        masked_logits = torch.where(action_mask, logits, torch.tensor(-1e8, device=logits.device))

        distribution.distribution.logits = masked_logits
        
        values = self.value_net(latent_vf)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy 