from .airsim_env import AirSimDroneEnv, TestEnv
import gymnasium as gym

# Register AirSim environment as a gym environment
gym.register(
    id='airsim-env-v0',
    entry_point='scripts:AirSimDroneEnv')


# Register AirSim environment as a gym environment
gym.register(
    id='test-env-v0',
    entry_point='scripts:TestEnv')
