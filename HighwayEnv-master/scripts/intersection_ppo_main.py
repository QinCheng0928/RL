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
import itertools
import functools

current_directory = os.getcwd()
print(current_directory)
# ============================================
#   tensorboard --logdir=log/intersection_ppo_MultiAgent/
# ============================================
# ==================================
#        Main script
# ==================================

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def train():
    n_cpu = 6
    batch_size = 128
    env = make_vec_env("intersection-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    # env = gym.make('intersection-v0')
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        n_steps= batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.99,
        verbose=2,
        clip_range=linear_schedule(0.2),
        tensorboard_log="log/intersection_ppo_fix_position/",
        # seed=2000,
        # device='cuda',  # 指定使用 GPU
        device='cpu', # 指定使用 CPU
    )
    # 检查使用的设备
    print("Device used:", model.policy.device)
    # Train the agent
    model.learn(total_timesteps=int(1e5))
    # Save the agent
    model.save("log/intersection_ppo_fix_position/model")


def get_action_table(count_vehicle, count_k, env):
    frames = int(
            env.config["simulation_frequency"] // env.config["policy_frequency"]
        )
    frames = 30
    # 保存动作的列表，先索引vehicle，在索引k
    dp = [["" for _ in range(count_k)] for _ in range(count_vehicle)]

    for k in range(count_k):
        for vehicle in range(count_vehicle):
            maxvalue = -np.inf
            for cur_action in env.ACTIONS.values():
                temp_env = deepcopy(env)
                controlled_vehicle = temp_env.controlled_vehicles[vehicle]
                if k == 0:
                    for i in temp_env.road.vehicles:
                        if not (controlled_vehicle is i):
                            i.action["acceleration"] = 0
                            i.action["steering"] = 0
                            i.speed = 0
                    controlled_vehicle.act(cur_action)
                else:
                    for i in range(count_vehicle):
                        if i != vehicle:
                            temp_env.controlled_vehicles[i].act(dp[i][k - 1])
                        else:
                            controlled_vehicle.act(cur_action)
                for _ in range(frames):
                    temp_env.road.act()
                    temp_env.road.step(1 / temp_env.config["simulation_frequency"])
                        
                predicted_reward = temp_env.my_reward(cur_action)
                if predicted_reward >= maxvalue:
                    maxvalue = predicted_reward
                    dp[vehicle][k] = cur_action
    return dp

def is_truncated(env) -> bool:
    # return env.time >= env.config["duration"]
    return env.time >= 1000

def is_terminated(env) -> bool:
    return (
        any(vehicle.crashed for vehicle in env.controlled_vehicles)
        or all(env.has_arrived(vehicle) for vehicle in env.controlled_vehicles)
        or (env.config["offroad_terminal"] and not env.vehicle.on_road)
    )
    

def evaluate():
    model = PPO.load(current_directory + "/log/intersection_ppo_fix_position/model")
    env = gym.make("intersection-v0", render_mode="human")
    ACTIONS_K = {i: list(comb) for i, comb in enumerate(itertools.product(list(range(4)), repeat=4))}

    for i in range(1):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            # obs, reward, done, truncated, info = env.step(action)
            env.time += 1 / env.config["policy_frequency"]
            
            dp = get_action_table(len(env.controlled_vehicles), 4, env)
            k_action = ACTIONS_K[int(action)]
            for i in range(len(env.controlled_vehicles)):
                env.controlled_vehicles[i].act(dp[i][k_action[i]])
            for vehicle in env.controlled_vehicles:
                if env.has_arrived(vehicle):
                    vehicle.speed = 0
                    vehicle.action["acceleration"] = 0
                    vehicle.action["steering"] = 0
                    
            env.road.step(1 / env.config["simulation_frequency"])
            for i in range(len(env.controlled_vehicles)):
                print("vehicle", i, ":", env.controlled_vehicles[i].speed)
            
            obs = env.observation_type.observe()
            done = is_terminated(env)
            truncated = is_truncated(env)
            if env.render_mode == "human":
                env.render()
            # for i in range(len(env.controlled_vehicles)):
            #     if env.has_arrived(env.controlled_vehicles[i]):
            #         env.controlled_vehicles[i].action["acceleration"] = 0
            #         env.controlled_vehicles[i].action["steering"] = 0
            #         env.controlled_vehicles[i].speed = 0
            env.render()
            

if __name__ == "__main__":
    istrain = False
    if istrain:
        print("Training...")
        train()
    else:
        print("evaluating...")
        evaluate()

