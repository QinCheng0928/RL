import copy
import numpy as np
import itertools
from copy import deepcopy

from sympy.physics.units import acceleration


class LevelK:
    # 返回型如[”SLOWER“，”SLOWER“，”SLOWER“，”SLOWER“]的列表
    def get_acceleration(self, env):
        count_k = 4                                     # 保存k的取值范围,表示 0 - 3
        count_vehicle = len(env.controlled_vehicles)    # 保存受控车的数量
        frames = 30                                      # 保存每次迭代的帧数

        best_actions = [0] * count_vehicle                                  # 保存动作列表，按照受控车的顺序
        dp = [["" for _ in range(count_k)] for _ in range(count_vehicle)]   # 保存迭代过程中的数据


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
        for i in range(count_vehicle):
            best_actions[i] = dp[i][env.controlled_vehicles[i].levelk]
        return best_actions
