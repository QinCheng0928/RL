import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from copy import deepcopy
import numpy as np
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
import highway_env
import warnings
warnings.filterwarnings("ignore")
import time

current_directory = os.getcwd()
print(current_directory)


def train():
    n_cpu = 6
    batch_size = 128
    # env = gym.make("intersection-multi-agent-v0",render_mode="rgb_array")
    # env = gym.make("intersection-multi-agent-v0")
    env = gym.make("intersection-v0")

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.99,
        verbose=2,
        tensorboard_log="log/single_intersection_ppo_1e5/",
    )

    # Train the agent
    model.learn(total_timesteps=int(1e5))
    # Save the agent
    model.save("log/single_intersection_ppo_1e5/model")

def evaluate():
    model = PPO.load(current_directory + "/log/single_intersection_ppo_1e5/model")
    # env = gym.make("intersection-v0",render_mode="human")
    env = gym.make("intersection-v0",render_mode="rgb_array")
    env.unwrapped.config.update({
        "controlled_vehicles": 4,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
            }
        },
        "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
            }
        }
    })
    
    for _ in range(5):
        obs, info = env.reset(seed=0)
        print(env.config)
        print(obs)
        done = truncated = False
        while not (done or truncated):
            action = tuple(model.predict(obs_i) for obs_i in obs)
            next_obs, reward, done, truncated, info = env.step(action)
            for obs_i, action_i, next_obs_i in zip(obs, action, next_obs):
              model.update(obs_i, action_i, next_obs_i, reward, info, done, truncated)
            obs = next_obs
            env.render()
            time.sleep(1)
            

if __name__ == "__main__":
    istrain = True
    if(istrain):
        print("training...")
        train()
    else:
        print("evaluating...")
        evaluate()


