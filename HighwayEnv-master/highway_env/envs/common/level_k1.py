# from highway_env.envs.common.action import DiscreteMetaAction
import csv
import numpy as np
import itertools
from copy import deepcopy

class LevelK:
    def reward(self, vehicle, other_vehicles):
        speed_reward = np.linalg.norm(vehicle.velocity)
        target_distance = np.linalg.norm(vehicle.position - vehicle.direction)
        target_reward = -target_distance

        collision_penalty = 0
        for other in other_vehicles:
            distance = np.linalg.norm(vehicle.position - other.position)
            if distance < 250:
                collision_penalty += (1 - distance / 250)
            elif 100 < distance <= 250:
                collision_penalty += (1 - distance / 250) * 20
            elif 60 < distance <= 100:
                collision_penalty += (1 - distance / 250) * 100
            elif distance < 60:
                collision_penalty += 100

        total_reward = speed_reward * 10 + target_reward - collision_penalty * 1000
        return total_reward


    # def update_state(self, env_copy, vehicle, time_step=0.5):
    #     if np.linalg.norm(vehicle.position - vehicle.direction) < 10:
    #         vehicle.position = vehicle.target_position
    #         vehicle.speed =0
    #     else:
    #         vehicle.velocity += vehicle * time_step
    #         vehicle.position += vehicle.velocity * time_step


    def get_acceleration(self, env_copy, vehicle, other_vehicles, k, time_step=0.5):
        other_vehicles_copy = deepcopy(other_vehicles)  # 创建副本
        if k == 0:
            for other_vehicle in other_vehicles_copy:
                other_vehicle.speed = 0
                other_vehicle.accelerate = 0
        elif k == 1:
            for other_vehicle in other_vehicles_copy:
                new_other_vehicles = [veh for veh in other_vehicles_copy if veh is not other_vehicle]  # 修正此处
                new_other_vehicles.append(vehicle)
                u2_k0 = self.get_acceleration(env_copy, other_vehicle, new_other_vehicles, 0, time_step)
                other_vehicle.act(u2_k0)
        elif k == 2:
            for other_vehicle in other_vehicles_copy:
                new_other_vehicles = [veh for veh in other_vehicles_copy if veh is not other_vehicle]  # 修正此处
                new_other_vehicles.append(vehicle)
                u2_k1 = self.get_acceleration(env_copy, other_vehicle, new_other_vehicles, 1, time_step)
                other_vehicle.act(u2_k1)
        elif k == 3:
            for other_vehicle in other_vehicles_copy:
                new_other_vehicles = [veh for veh in other_vehicles_copy if veh is not other_vehicle]  # 修正此处
                new_other_vehicles.append(vehicle)
                u3_k1 = self.get_acceleration(env_copy, other_vehicle, new_other_vehicles, 2, time_step)
                other_vehicle.act(u3_k1)

        # 调试输出
        # print("Before calling choose_best_action:")
        # print(f"Vehicle type: {type(vehicle)}, Vehicle: {vehicle}")
        # print(f"Other vehicles type: {type(other_vehicles_copy)}, Other vehicles: {other_vehicles_copy}")

        best_action = self.choose_best_action(vehicle, other_vehicles_copy, time_step)
        vehicle.accelerate = best_action
        return best_action


    def choose_best_action(self, vehicle, other_vehicles, time_step=0.5):
        # print("Inside choose_best_action:")
        # print(f"Vehicle type: {type(vehicle)}, Vehicle: {vehicle}")
        # print(f"Other vehicles type: {type(other_vehicles)}, Other vehicles: {other_vehicles}")

        actions = ["SLOWER", "IDLE", "FASTER"]
        best_action = None
        best_reward = -np.inf

        for action in actions:
            original =deepcopy(vehicle)

            vehicle.act(action)

            # ????????
            # predicted_reward = sum(self.reward(vehicle,other) for other in other_vehicles)
            predicted_reward = self.reward(vehicle, other_vehicles)

            if predicted_reward > best_reward:
                best_reward = predicted_reward
                best_action = action

            vehicle = original

        return best_action
