import numpy as np
import gymnasium as gym
from copy import deepcopy

# 定义常数
count_vehicle = 4
count_k = 4

def action_table(env):
    # 保存动作的列表，先索引vehicle，在索引k
    dp = [["" for _ in range(count_k)] for _ in range(count_vehicle)]

    for k in range(count_k):
        for vehicle in range(count_vehicle):
            maxvalue = -np.inf
            for cur_action in env.ACTIONS.values():
                temp_env = deepcopy(env)
                if k == 0:
                    for i in temp_env.road.vehicles:
                        if not (temp_env.controlled_vehicles[vehicle] is i):
                            i.action["acceleration"] = 0
                            i.action["steering"] = 0
                            i.speed = 0
                        else:
                            temp_env.controlled_vehicles[vehicle].act(cur_action)
                    temp_env.road.act()
                    temp_env.road.step(1 / temp_env.config["simulation_frequency"])
                    predicted_reward = temp_env._reward(cur_action)
                    if predicted_reward > maxvalue:
                        maxvalue = predicted_reward
                        dp[vehicle][k] = cur_action
                else:
                    for i in range(count_vehicle):
                        if i != vehicle:
                            temp_env.controlled_vehicles[i].act(dp[i][k - 1])
                        else:
                            temp_env.controlled_vehicles[vehicle].act(cur_action)
                    temp_env.road.act()
                    temp_env.road.step(1 / temp_env.config["simulation_frequency"])
                    predicted_reward = temp_env._reward(cur_action)
                    if predicted_reward > maxvalue:
                        maxvalue = predicted_reward
                        dp[vehicle][k] = cur_action



if __name__ == "__mian__":
    # 创建环境
    env = gym.make("intersection-v0", render_mode="human")
    # 获得车辆不同action的列表
    best_actions_table = action_table(env)



