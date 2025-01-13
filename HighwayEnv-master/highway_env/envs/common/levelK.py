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

    def reward(self, vehicle):
        speed_reward = vehicle.speed
        total_reward = speed_reward
        return total_reward


    def get_acceleration(self, env_copy, vehicle_list, time_step=0.5):

        best_actions = [0] * len(vehicle_list)
        vehicle_list_copy = deepcopy(vehicle_list)
        for i in range(len(vehicle_list)):
            index=-1
            value_min=10

            for j in range(4-i):
                if value_min>vehicle_list[j].levelk:
                    value_min=vehicle_list[j].levelk
                    index=j

            vehicle=vehicle_list[index]

            other_vehicles = [v for v in vehicle_list if v != vehicle]

            for other_vehicle in other_vehicles:
                if other_vehicle.levelk >= vehicle.levelk:
                   other_vehicle.speed = 0
                   other_vehicle.accelerate = 0

            best_action = self.choose_best_action(vehicle, other_vehicles, time_step)
            vehicle.act(best_action)
            del vehicle_list_copy[index]
            best_actions[index]=best_action

        return best_actions


    def choose_best_action(self, vehicle, other_vehicles, time_step=0.5):
        # print("Inside choose_best_action:")
        # print(f"Vehicle type: {type(vehicle)}, Vehicle: {vehicle}")
        # print(f"Other vehicles type: {type(other_vehicles)}, Other vehicles: {other_vehicles}")

        best_action = None
        best_reward = -np.inf

        for action in self.actions:
            original =deepcopy(vehicle)

            vehicle.act(action)

            predicted_reward = self.reward(vehicle)

            if predicted_reward > best_reward:
                best_reward = predicted_reward
                best_action = action

            vehicle = original

        return best_action
