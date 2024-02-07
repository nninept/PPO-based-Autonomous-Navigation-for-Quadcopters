import numpy as np
import d3rlpy
import torch
import torch.nn as nn
import gymnasium as gym
import yaml
import dataclasses
import d3rlpy.algos as algos
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.metrics import EnvironmentEvaluator
import scripts

with open('scripts/config.yml', 'r') as f:
    env_config = yaml.safe_load(f)
    
# Create a DummyVecEnv
env = gym.make("airsim-env-v0",
        ip_address="127.0.0.1", 
        image_shape=(3,50,50),
        env_config=env_config["TrainEnv"])

np_data = np.load('./offlinerl_dataset.npz')

dataset = d3rlpy.dataset.MDPDataset(
    observations=np_data['observations'],
    # actions=actions,
    actions=np_data['actions'],
    rewards=np_data['rewards'],
    terminals=np_data['terminals'],
    action_size=9
)
# print(dataset.episodes)

class CustomCNN(nn.Module):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim):
        super().__init__()
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        
        self.cnn = nn.Sequential(
            #50 50
            nn.Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),

            #25 25
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),

            #12 12
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),

            #10 10
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),

            #8 8
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),

            nn.Flatten(start_dim=1, end_dim=-1)
        )
        
        observation_space_box = gym.spaces.Box(low=0, high=255, shape=observation_space, dtype=np.uint8)
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space_box.sample()[None]).float()
            ).shape[1]
            
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


@dataclasses.dataclass()
class CustomEncoderFactory(d3rlpy.models.EncoderFactory):
    feature_size: int

    def create(self, observation_shape):
        return CustomCNN(observation_shape, self.feature_size)

    @staticmethod
    def get_type() -> str:
        return "custom"
# if you don't use GPU, set device=None instead.
    

# model = algos.DQNConfig(encoder_factory=CustomEncoderFactory(256)).create(device="cuda:0")
model = algos.DQNConfig().create(device="cuda:0")

# initialize neural networks with the given observation shape and action size.
# this is not necessary when you directly call fit or fit_online method.
model.build_with_dataset(dataset)

# calculate metrics with training dataset
td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)

# set environment in scorer function
env_evaluator = EnvironmentEvaluator(env)

# evaluate algorithm on the environment
rewards = env_evaluator(model, dataset=None)

model.fit(
    dataset,
    n_steps=10000,
    evaluators={
        'td_error': td_error_evaluator,
        'environment': env_evaluator,
    },
)

model.save('./saved_policy/offline_model.d3')