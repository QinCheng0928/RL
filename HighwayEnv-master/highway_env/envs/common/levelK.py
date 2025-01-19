# from highway_env.envs.common.action import DiscreteMetaAction
import copy
import numpy as np
import itertools
from copy import deepcopy

class LevelK:
    def __init__(self):
        pass
        self.actionslist = ["SLOWER", "IDLE", "FASTER"]

    def get_acceleration(self, env, vehicle_list, time_step=0.5):
        # 保存动作列表，按照受控车的顺序
        best_actions = [0] * len(vehicle_list)
        # 标记每辆车是否已经选择动作
        selected = [False] * len(vehicle_list)
        best_reward = 0
        
        # 查找K值最小的车辆
        for i in range(len(vehicle_list)):
            index = -1
            minvalue = np.inf
            
            for j in range(4):
                if not selected[j] and minvalue > vehicle_list[j].levelk:
                    minvalue = vehicle_list[j].levelk
                    index = j
                    
            vehicle = vehicle_list[index]
            selected[index] = True
            other_vehicles = [v for j, v in enumerate(vehicle_list) if j != index]

            # 假设优先级高的车辆会让优先级低的车辆停止
            for other_vehicle in other_vehicles:
                if other_vehicle.levelk >= vehicle.levelk:
                    other_vehicle.speed = 0
                    other_vehicle.action["acceleration"] = 0
                    other_vehicle.action["steering"] = 0

            # 选择最优动作
            best_action,best_vehicle_reward = self.choose_best_action(env, vehicle)
            vehicle.act(best_action)
            env.road.step(1 / env.config["simulation_frequency"])
            best_actions[index] = best_action
            best_reward += best_vehicle_reward

        return best_actions,best_reward


    def choose_best_action(self, env, vehicle):      
        best_action = None
        best_reward = -np.inf

        for action in self.actionslist:
            copy_env = copy.deepcopy(env)
            copy_vehicle = copy.deepcopy(vehicle)
            # 更新该车辆（vehicle）的切向加速度和径向加速度
            copy_vehicle.act(action)
            # 依据环境中每辆车的加速度更新车辆的状态
            copy_env.road.step(1 / copy_env.config["simulation_frequency"])

            # 计算vehicle执行动作的奖励
            # predicted_reward = copy_env._agent_reward(action,copy_vehicle)
            predicted_reward = copy_env._reward(action)

            if predicted_reward > best_reward:
                best_reward = predicted_reward
                best_action = action


        return best_action, best_reward
