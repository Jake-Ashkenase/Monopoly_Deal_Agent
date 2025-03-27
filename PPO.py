from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Game_Representation_RL import MonopolyDealEnv
import numpy as np

# Create and wrap the environment
def make_env():
    return MonopolyDealEnv()

env = DummyVecEnv([make_env])

# Initialize the PPO agent
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./monopoly_deal_tensorboard/"
)

# Train the agent
TOTAL_TIMESTEPS = 1_000
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Save the trained model
model.save("monopoly_deal_ppo")

# Test the trained agent
# obs = env.reset()
# episodes = 10
# for episode in range(episodes):
#     done = False
#     total_reward = 0
#     while not done:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
#         total_reward += reward
#     print(f"Episode {episode + 1}: Total Reward = {total_reward}")