from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Game_Representation_RL import MonopolyDealEnv
from custom_policy import CustomMaskedPolicy
import numpy as np
import torch
import torch.nn as nn

def make_env():
    env = MonopolyDealEnv()
    return env

def setup_model():
    # Use DummyVecEnv but ensure it preserves our Dict observation space
    env = DummyVecEnv([make_env])

    # Initialize the PPO agent with our custom policy
    model = PPO(
        CustomMaskedPolicy,  
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Add entropy to encourage exploration
        tensorboard_log="./monopoly_deal_tensorboard/",
        policy_kwargs=dict(
            # Much larger networks to capture complex game strategies
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            activation_fn=nn.ReLU
        )
    )
    return model, env

def train_model(total_timesteps=1_000):
    """
    Train the PPO model and save it
    
    Args:
        total_timesteps (int): Number of timesteps to train for
    """
    model, _ = setup_model()
    
    # Train the agent
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model.save("monopoly_deal_ppo")
    print("Model trained and saved!")

def test_model(episodes=100, model_path="monopoly_deal_ppo"):
    """
    Load and test a trained model
    
    Args:
        episodes (int): Number of episodes to test
        model_path (str): Path to the saved model
    """
    # Load the trained model
    model = PPO.load(model_path, env=DummyVecEnv([make_env]))
    env = model.get_env()
    
    # Test the trained agent
    obs = env.reset()
    total_rewards = []
    wins = 0
    losses = 0
    draws = 0
    
    for episode in range(episodes):
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if info[0]['Game_status'] == 'Agent wins':
                wins += 1
            elif info[0]['Game_status'] == 'Opponent wins':
                losses += 1
            elif info[0]['Game_status'] == 'Draw':
                draws += 1

        total_rewards.append(total_reward)
    
    # Print summary statistics
    print("\nTesting Summary:")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Standard Deviation: {np.std(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"\nWin Rate: {wins/episodes:.2%}")
    print(f"Loss Rate: {losses/episodes:.2%}")
    print(f"Draw Rate: {draws/episodes:.2%}")

if __name__ == "__main__":
    train_model(total_timesteps=100_000)
    test_model(episodes=200)
