import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

from copy import deepcopy
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
import highway_env
from highway_env.envs.intersection_env import IntersectionEnv
import warnings
warnings.filterwarnings("ignore")


def is_truncated(env) -> bool:
    # return env.time >= env.config["duration"]
    return env.time >= 1000

def is_terminated(env) -> bool:
    return (
        any(vehicle.crashed for vehicle in env.controlled_vehicles)
        or all(env.has_arrived(vehicle) for vehicle in env.controlled_vehicles)
        or (env.config["offroad_terminal"] and not env.vehicle.on_road)
    )
    
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

if __name__ == '__main__':
    # 创建环境
    env = gym.make("intersection-v0", render_mode="human")
    
    # 定义常数
    count_vehicle = len(env.controlled_vehicles)
    count_k = 4
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):        
        best_action_table = get_action_table(count_vehicle, count_k, env)
        # print("best_action_table:", best_action_table)
        
        maxvalue = -np.inf
        best_action = [0] * count_vehicle
        for i in range(count_k):
            for j in range(count_k):
                for k in range(count_k):
                    for l in range(count_k):
                        temp_env = deepcopy(env)
                        temp_reward = temp_env.my_reward([best_action_table[0][i], best_action_table[1][j], best_action_table[2][k], best_action_table[3][l]])
                        if temp_reward > maxvalue:
                            maxvalue = temp_reward
                            best_action = [best_action_table[0][i], best_action_table[1][j], best_action_table[2][k], best_action_table[3][l]]
                            
        print("best_action:", best_action)
        print("maxvalue:", maxvalue)
        
        for i in range(count_vehicle):
            env.controlled_vehicles[i].act(best_action[i])
            
        # 控制到达目的地的车辆使其静止
        for i in range(count_vehicle):
            if env.has_arrived(env.controlled_vehicles[i]):
                env.controlled_vehicles[i].action["acceleration"] = 0
                env.controlled_vehicles[i].action["steering"] = 0
                env.controlled_vehicles[i].speed = 0
                
        env.road.step(1 / env.config["simulation_frequency"])
        for i in range(count_vehicle):
            print("vehicle", i, ":", env.controlled_vehicles[i].speed)
        terminated = is_terminated(env)
        truncated = is_truncated(env)
        env.render() 
        env.time += 1 / env.config["simulation_frequency"]
        

                    








