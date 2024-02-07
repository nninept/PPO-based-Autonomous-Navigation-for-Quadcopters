import gymnasium as gym
import yaml
import d3rlpy
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import scripts

# Get train environment configs
with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)


# Create a DummyVecEnv
env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "test-env-v0",
        ip_address="127.0.0.1", 
        image_shape=(3,50,50),
        env_config=env_config["TrainEnv"]
        )
)])

model = d3rlpy.load_learnable("./saved_policy/offline_model.d3")

obs = env.reset()
for i in range(1000):
    action = model.predict(obs)
    obs, _, dones, info = env.step(action)
