# from highway_env.envs.common.action import DiscreteMetaAction
import csv
import numpy as np
import itertools
from copy import deepcopy

class LevelK:
    def __init__(self):
        self.actions = ["SLOWER", "IDLE", "FASTER"]

    def get_acceleration(self, env_copy, vehicle_list, time_step=0.5):
        # 通过 levelk 排序，避免在每次循环中寻找最小值
        vehicle_list_sorted = sorted(vehicle_list, key=lambda v: v.levelk)
        best_actions = [0] * len(vehicle_list)

        # 遍历车辆，按照排序依次选择动作
        for i, vehicle in enumerate(vehicle_list_sorted):
            # 获取其他车辆
            other_vehicles = vehicle_list_sorted[:i] + vehicle_list_sorted[i + 1:]

            # 假设优先级高的车辆会让优先级低的车辆停止
            for other_vehicle in other_vehicles:
                if other_vehicle.levelk >= vehicle.levelk:
                    other_vehicle.speed = 0
                    other_vehicle.action["acceleration"] = 0
                    other_vehicle.action["steering"] = 0

            # 选择最优动作
            best_action,best_reward = self.choose_best_action(env_copy, vehicle, other_vehicles, time_step)
            vehicle.act(best_action)
            env_copy.road.step(1 / env_copy.config["simulation_frequency"])
            best_actions[i] = best_action

        return best_actions,best_reward


    def choose_best_action(self, env_copy, vehicle, other_vehicles, time_step=0.5):
        # print("Inside choose_best_action:")
        # print(f"Vehicle type: {type(vehicle)}, Vehicle: {vehicle}")
        # print(f"Other vehicles type: {type(other_vehicles)}, Other vehicles: {other_vehicles}")

        best_action = None
        best_reward = -np.inf

        for action in self.actions:
            original_speed = vehicle.speed
            original_accelerate = vehicle.action["acceleration"]
            original_steering = vehicle.action["steering"]

            # 更新该车辆（vehicle）的切向加速度和径向加速度
            vehicle.act(action)
            # 依据环境中每辆车的加速度更新车辆的状态
            env_copy.road.step(1 / env_copy.config["simulation_frequency"])

            # 计算vehicle执行动作的奖励
            predicted_reward = env_copy._agent_reward(action,vehicle)

            if predicted_reward > best_reward:
                best_reward = predicted_reward
                best_action = action

            # 恢复车辆状态
            vehicle.speed = original_speed
            vehicle.action["acceleration"] = original_accelerate
            vehicle.action["steering"] = original_steering

        return best_action, best_reward
