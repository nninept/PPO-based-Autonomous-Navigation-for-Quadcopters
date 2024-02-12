import gym
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import matplotlib.pyplot as plt
import numpy as np

# Get train environment configs
with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)

# Create a DummyVecEnv
env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "scripts:airsim-env-v0", 
        ip_address="127.0.0.1", 
        image_shape=(50,50,3),
        env_config=env_config["TrainEnv"]
    )
)])

# Wrap env as VecTransposeImage (Channel last to channel first)
env = VecTransposeImage(env)

# Load an existing model
model = PPO.load(env=env, path="saved_policy\ppo_navigation_policy")
# model = PPO.load(env=env, path="saved_policy\ppo_navigation_policy202")

# Run the trained policy

data_observations = []
data_actions = []
data_rewards = []
data_terminals = []


epi_observations = []
epi_actions = []
epi_rewards = []
epi_terminals = []


obs = env.reset()
while len(data_observations) < 10000:
    epi_observations.append(obs)

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, info = env.step(action)
    
    action_vector = np.zeros((1, env.action_space.n))
    action_vector[:,action] = 1

    epi_actions.append(action)
    epi_rewards.append(reward)
    epi_terminals.append(dones)

    if(dones):
        if(info[0]["successfull"]):
            data_observations.append(np.concatenate(epi_observations))
            data_actions.append(np.concatenate(epi_actions))
            data_rewards.append(np.array(epi_rewards))
            data_terminals.append(np.array(epi_terminals))

        epi_observations = []
        epi_actions = []
        epi_rewards = []
        epi_terminals = []

        obs = env.reset()

        if(len(data_observations) % 50 == 0):
            print(f"{len(data_observations)} Done!")

print("Total Episode : ", len(data_observations))

data_observations = np.concatenate(data_observations)
data_observations = np.transpose(data_observations, (0,2,3,1))
print()
data_actions = np.concatenate(data_actions)
data_rewards = np.concatenate(data_rewards)
data_terminals = np.concatenate(data_terminals)


with open('offlinerl_datasets.npz', 'wb') as f:
    np.savez(f, 
             observations=data_observations,
             actions=data_actions,
             rewards=data_rewards,
             terminals=data_terminals)