# from highway_env.envs.common.action import DiscreteMetaAction
import csv
import numpy as np
import itertools
from copy import deepcopy

class LevelK:
    def __init__(self):
        self.action_map = {
            "SLOWER": -5,
            "IDLE": 0,
            "FASTER": 5
        }
        self.actions = ["SLOWER", "IDLE", "FASTER"]

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


    def get_acceleration(self, env_copy, vehicle, other_vehicles, k, time_step=0.5):
        other_vehicles_copy = deepcopy(other_vehicles) 
        if k == 0:
            for other_vehicle in other_vehicles_copy:
                other_vehicle.speed = 0
                other_vehicle.accelerate = 0
        elif k >= 1 and k <= 3:
            for other_vehicle in other_vehicles_copy:
                new_other_vehicles = [veh for veh in other_vehicles_copy if veh is not other_vehicle] 
                new_other_vehicles.append(vehicle)
                temp_action = self.get_acceleration(env_copy, other_vehicle, new_other_vehicles, k-1, time_step)
                other_vehicle.act(temp_action)
        else:
            print("Error: k should be 0, 1, 2 or 3.")

        best_action = self.choose_best_action(vehicle, other_vehicles_copy, time_step)
        vehicle.accelerate = self.action_map[best_action]
        return best_action


    def choose_best_action(self, vehicle, other_vehicles, time_step=0.5):
        # print("Inside choose_best_action:")
        # print(f"Vehicle type: {type(vehicle)}, Vehicle: {vehicle}")
        # print(f"Other vehicles type: {type(other_vehicles)}, Other vehicles: {other_vehicles}")

        best_action = None
        best_reward = -np.inf

        for action in self.actions:
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
