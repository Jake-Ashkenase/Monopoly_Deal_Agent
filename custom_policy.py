import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.ppo.policies import ActorCriticPolicy
import gym
from typing import Dict, List, Tuple, Type, Union, Optional
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
            10 +  # Card mask
            10    # Property mask
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
        # Normalize values to [0,1] range for more stable training
        agent_hand = observations["Agent hand"].float().reshape(batch_size, -1)[:, :10] / 18.0  # Cards range 0-18
        agent_board = observations["Agent Board"].float().reshape(batch_size, -1)[:, :10] / 10.0  # Properties range 0-10
        opponent_board = observations["Opponent Board"].float().reshape(batch_size, -1)[:, :10] / 10.0
        agent_cash = observations["Agent Cash"].float().reshape(batch_size, -1)[:, :6] / 6.0  # Cash range 0-6
        opponent_cash = observations["Opponent Cash"].float().reshape(batch_size, -1)[:, :6] / 6.0
        turn = observations["Turn"].float().reshape(batch_size, -1)[:, :1] / 5.0  # Turn range 0-5
        
        card_mask = observations["card_mask"].float().reshape(batch_size, -1)[:, :10]
        property_mask = observations["property_mask"].float().reshape(batch_size, -1)[:, :10]

        # Concatenate all features along the feature dimension (dim=1)
        combined = torch.cat([
            agent_hand,
            agent_board,
            opponent_board,
            agent_cash,
            opponent_cash,
            turn,
            card_mask,
            property_mask
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
        
        # Create separate prediction networks for card selection and property selection
        self.card_action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, 10)  # 10 card options
        self.property_action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, 10)  # 10 property options

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

        # Get card action logits
        card_logits = self.card_action_net(latent_pi)
        
        # Get property action logits
        property_logits = self.property_action_net(latent_pi)
        
        # Get the action masks and convert to bool
        card_mask = obs["card_mask"].bool()
        property_mask = obs["property_mask"].bool()
        
        # Handle the case where no properties are available to steal
        # For each batch item, check if all properties are invalid
        batch_size = property_mask.shape[0]
        all_invalid = ~torch.any(property_mask, dim=1)  # True if all properties are invalid
        
        # Create a default valid property mask for cases where all are invalid
        # This prevents trying to sample from an all-zero distribution
        default_property_mask = property_mask.clone()
        for i in range(batch_size):
            if all_invalid[i]:
                # If all properties are invalid, make the first one valid
                # This is just for the policy - the environment will handle it correctly
                default_property_mask[i, 0] = True
        
        # Mask invalid card actions with negative infinity
        masked_card_logits = card_logits.masked_fill(~card_mask, float('-inf'))
        
        # Use the modified property mask that has at least one valid option
        masked_property_logits = property_logits.masked_fill(~default_property_mask, float('-inf'))
        
        # Always use deterministic selection - take argmax of masked logits
        card_actions = torch.argmax(masked_card_logits, dim=1).long()
        property_actions = torch.argmax(masked_property_logits, dim=1).long()
        
        # Combine actions into a single multi-discrete action
        actions = torch.stack([card_actions, property_actions], dim=1)
        
        # Calculate log probabilities for the chosen actions
        # Use softmax to get probabilities
        card_probs = torch.softmax(masked_card_logits, dim=1)
        property_probs = torch.softmax(masked_property_logits, dim=1)
        
        # Get probability of selected actions
        batch_indices = torch.arange(card_actions.size(0), device=card_actions.device).long()
        card_prob = card_probs[batch_indices, card_actions]
        
        # For property probability, if all properties are invalid, use 1.0 (log prob = 0)
        # This means "the probability of the default action when no choice matters is 1"
        property_prob = torch.ones_like(card_prob)
        valid_property_selections = ~all_invalid
        property_prob[valid_property_selections] = property_probs[valid_property_selections, property_actions[valid_property_selections]]
        
        # Combined log prob is sum of individual log probs (only consider card prob when all properties invalid)
        log_prob = torch.log(card_prob)
        log_prob[valid_property_selections] += torch.log(property_prob[valid_property_selections])

        return actions, values, log_prob

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions (MultiDiscrete)
        :return: estimated value, log likelihood of taking those actions, entropy of the action distribution
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Split actions into card and property components
        card_actions = actions[:, 0].long()  
        property_actions = actions[:, 1].long()  
        
        # Get card action logits
        card_logits = self.card_action_net(latent_pi)
        
        # Get property action logits
        property_logits = self.property_action_net(latent_pi)
        
        # Apply action masks
        card_mask = obs["card_mask"].bool()
        property_mask = obs["property_mask"].bool()
        
        # Check if all properties are invalid for each batch item
        batch_size = property_mask.shape[0]
        all_invalid = ~torch.any(property_mask, dim=1)  # True if all properties are invalid
        
        # Create default property mask for cases where all are invalid
        default_property_mask = property_mask.clone()
        for i in range(batch_size):
            if all_invalid[i]:
                default_property_mask[i, 0] = True
        
        masked_card_logits = card_logits.masked_fill(~card_mask, float('-inf'))
        masked_property_logits = property_logits.masked_fill(~default_property_mask, float('-inf'))
        
        # Get probabilities using softmax
        card_probs = torch.softmax(masked_card_logits, dim=1)
        property_probs = torch.softmax(masked_property_logits, dim=1)
        
        # Calculate entropies - only count property entropy when there are valid properties
        card_entropy = -torch.sum(card_probs * torch.log(card_probs + 1e-8), dim=1)
        property_entropy = torch.zeros_like(card_entropy)
        valid_property_selections = ~all_invalid
        if torch.any(valid_property_selections):
            property_entropy[valid_property_selections] = -torch.sum(
                property_probs[valid_property_selections] * 
                torch.log(property_probs[valid_property_selections] + 1e-8), 
                dim=1
            )
        
        # Total entropy is sum of card entropy plus property entropy (when valid)
        entropy = card_entropy + property_entropy
        
        # Get log probabilities of selected actions
        batch_indices = torch.arange(card_actions.size(0), device=card_actions.device).long()
        card_prob = card_probs[batch_indices, card_actions]
        
        # For property probability, if all properties are invalid, use 1.0 (log prob = 0)
        property_prob = torch.ones_like(card_prob)
        if torch.any(valid_property_selections):
            property_prob[valid_property_selections] = property_probs[
                valid_property_selections, 
                property_actions[valid_property_selections]
            ]
        
        # Combined log prob is sum of individual log probs (only when properties valid)
        log_prob = torch.log(card_prob + 1e-8)
        log_prob[valid_property_selections] += torch.log(property_prob[valid_property_selections] + 1e-8)
        
        values = self.value_net(latent_vf)

        return values, log_prob, entropy

    def predict(
        self,
        observation: Dict[str, np.ndarray],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        
        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state (used in recurrent policies)
        """
        # Convert numpy arrays to torch tensors
        obs_tensor = {key: torch.as_tensor(val).to(self.device) for key, val in observation.items()}
        
        with torch.no_grad():
            # Get features and latent policy
            features = self.extract_features(obs_tensor)
            latent_pi, _ = self.mlp_extractor(features)
            
            # Get card and property logits
            card_logits = self.card_action_net(latent_pi)
            property_logits = self.property_action_net(latent_pi)
            
            # Get action masks and apply them
            card_mask = obs_tensor["card_mask"].bool()
            property_mask = obs_tensor["property_mask"].bool()
            
            # Handle case where no properties are available
            all_invalid = ~torch.any(property_mask, dim=1)
            default_property_mask = property_mask.clone()
            for i in range(len(all_invalid)):
                if all_invalid[i]:
                    default_property_mask[i, 0] = True
            
            masked_card_logits = card_logits.masked_fill(~card_mask, float('-inf'))
            masked_property_logits = property_logits.masked_fill(~default_property_mask, float('-inf'))
            
            # Get actions (always use deterministic for this environment)
            card_actions = torch.argmax(masked_card_logits, dim=-1).long()
            property_actions = torch.argmax(masked_property_logits, dim=-1).long()
            
            # Combine into one multidiscrete action
            actions = torch.stack([card_actions, property_actions], dim=1)
            
        # Convert back to numpy
        actions = actions.cpu().numpy()
        
        return actions, state 