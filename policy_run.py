import gymnasium as gym
import yaml

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

# Load an existing model
model = PPO.load(env=env, path="saved_policy\ppo_navigation_policy")

# Run the trained policy
obs = env.reset()
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, dones, info = env.step(action)
    
    # print(obs.shape)
    # new_obs = obs.squeeze(0)
    # new_obs = np.transpose(new_obs, (1,2,0))
    # plt.imshow(new_obs)
    # plt.show()
    
