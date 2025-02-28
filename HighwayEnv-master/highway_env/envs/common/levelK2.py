import copy
import numpy as np
import itertools
from copy import deepcopy

from sympy.physics.units import acceleration
   
def get_action_table(count_vehicle, count_k, env):
    frames = 30
    # 保存动作的列表，先索引vehicle，在索引k
    dp = [[{"reward":-np.inf,"action":""} for _ in range(count_k)] for _ in range(count_vehicle)]

    for k in range(count_k):
        for vehicle in range(count_vehicle):
            maxvalue = -np.inf
            for cur_action in env.ACTIONS.values():
                temp_env = deepcopy(env)
                controlled_vehicle = temp_env.controlled_vehicles[vehicle]
                if k == 0:
                    for i in temp_env.controlled_vehicles:
                        if not (controlled_vehicle is i):
                            i.action["acceleration"] = 0
                            i.action["steering"] = 0
                            i.speed = 0
                    controlled_vehicle.act(cur_action)
                    temp_env.road.step(1 / temp_env.config["simulation_frequency"])
                else:
                    for i in range(count_vehicle):
                        if i != vehicle:
                            temp_env.controlled_vehicles[i].act(dp[i][k - 1]["action"])
                        else:
                            controlled_vehicle.act(cur_action)
                            temp_env.road.step(1 / temp_env.config["simulation_frequency"])
                            
                predicted_reward = 0.0
                for _ in range(frames):
                    temp_env.road.act()
                    temp_env.road.step(1 / temp_env.config["simulation_frequency"])
                    predicted_reward += temp_env.my_reward()
                # predicted_reward = temp_env.my_reward()        
                if predicted_reward > maxvalue:
                    maxvalue = predicted_reward
                    dp[vehicle][k]["action"] = cur_action
                    dp[vehicle][k]["reward"] = predicted_reward

    return dp

class LevelK:
    # 返回型如[”SLOWER“，”SLOWER“，”SLOWER“，”SLOWER“]的列表
    def get_acceleration(self, env):
        count_k = 4                                     # 保存k的取值范围,表示 0 - 3
        count_vehicle = len(env.controlled_vehicles)    # 保存受控车的数量

        best_actions = [0] * count_vehicle                                  # 保存动作列表，按照受控车的顺序
        
        dp = get_action_table(count_vehicle, count_k, env)
        
        for i in range(count_vehicle):
            best_actions[i] = dp[i][env.controlled_vehicles[i].levelk]["action"]
        return best_actions
