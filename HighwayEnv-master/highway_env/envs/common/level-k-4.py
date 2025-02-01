import csv
import numpy as np
import itertools

class Vehicle:
    def __init__(self, position, velocity, accelerate, target_position, color, name):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.accelerate = np.array(accelerate, dtype=np.float64)
        self.target_position = np.array(target_position, dtype=np.float64)
        self.color = color
        self.name = name
        self.trajectory = [self.position.copy()]

    
          
    def reward(vehicle, vehicles):
        other_vehicles=[]
        for j in vehicles:
            if j.name!=vehicle.name:
                other_vehicles.append(j)
            else:
                continue
        speed_reward = np.linalg.norm(vehicle.velocity)
        target_distance = np.linalg.norm(vehicle.position - vehicle.target_position)
        target_reward = -target_distance * 1
    
        # 每个车辆之间的距离以及碰撞惩罚
        collision_penalty = 0
        for other_vehicle in other_vehicles:
            distance = np.linalg.norm(vehicle.position - other_vehicle.position)
            if distance < 10:
                collision_penalty += (1 - distance / 10)
    
        total_reward = speed_reward + target_reward - collision_penalty * 1000
        return total_reward


    def update_state(vehicle,actions,time_step=0.5):
      action=np.array([0, 0])
      vehicle.accelerate=action
      if np.linalg.norm(vehicle.position - vehicle.target_position) < 10:
          vehicle.position=vehicle.target_position
          vehicle.velocity=[0,0]
          
      else:
          vehicle.velocity += vehicle.accelerate * time_step
          vehicle.position += vehicle.velocity * time_step

          # 确保B车的X坐标不变
          if vehicle.name == "B" or vehicle.name == "D":
              vehicle.position[0] = vehicle.target_position[0]
              
          if vehicle.name == "A"or vehicle.name=="C":
              # 只更新X坐标
            vehicle.position[1] = vehicle.target_position[1]

      vehicle.trajectory.append(vehicle.position.copy())

    def get_acceleration(vehicle, vehicles, k, time_step=0.5):
        other_vehicles=[]
        for j in vehicles:
            if j.name!=vehicle.name:
                other_vehicles.append(j)
            else:
                continue
        if k == 0:
            for other_vehicle in other_vehicles:
                other_vehicle.velocity = np.array([0, 0],dtype=np.float64)
                other_vehicle.action = np.array([0, 0],dtype=np.float64)
        elif k == 1:
            for other_vehicle in other_vehicles:
                u2_k0 = Vehicle.get_acceleration(other_vehicle,vehicles, 0, time_step)
                other_vehicle.action = u2_k0
                Vehicle.update_state(other_vehicle,u2_k0)
        elif k == 2:
            for other_vehicle in other_vehicles:
                u2_k1 = other_vehicle.get_acceleration(other_vehicles, 1, time_step)
                other_vehicle.accelerate = u2_k1
                Vehicle.update_state(other_vehicle,u2_k1)

        best_action  = Vehicle.choose_best_action(vehicle,vehicles,  time_step)
        return best_action

    def choose_best_action(vehicle, vehicles, time_step=0.5):
        actions = {
            "A": [np.array([0, 0]), np.array([0.5, 0]), np.array([-0.5, 0])],
            "B": [np.array([0, 0]), np.array([0, 0.5]), np.array([0, -0.5])],
            "C": [np.array([0, 0]), np.array([0.5, 0]), np.array([-0.5, 0])],
            "D": [np.array([0, 0]), np.array([0, 0.5]), np.array([0, -0.5])]
        }
    
        best_action = None
        best_reward = 0
        predicted_reward=0
        for action in actions[vehicle.name]:
            original_position = vehicle.position.copy()
            original_velocity = vehicle.velocity.copy()
    
            # 更新状态并计算奖励
            Vehicle.update_state(vehicle, action, time_step)
            predicted_reward += Vehicle.reward(vehicle, vehicles)
            print(f"Action: {action}, Predicted reward: {predicted_reward}")  # 调试输出
    
            # 更新最佳动作
            if predicted_reward > best_reward:
                best_reward = predicted_reward
                best_action = action
    
            # 恢复状态
            vehicle.position = original_position
            vehicle.velocity = original_velocity

        print(f"Chosen best action: {best_action}, Reward: {best_reward}")  # 调试输出
        return best_action
    


def predict_best_actions(vehicles, time_step=0.5):
    best_actions = [None] * len(vehicles)
    best_total_reward = -np.inf
    total_reward = -np.inf
    
    # 备份原始状态
    original_positions = [i.position.copy() for i in vehicles]
    original_velocities = [i.velocity.copy() for i in vehicles]
    for a in range(3):
        for b in  range(3):
            for c in range(3):
                for d in range(3):
                      a1=Vehicle.get_acceleration(vehicles[0], vehicles, a)
                      b1=Vehicle.get_acceleration(vehicles[1], vehicles, b)
                      c1=Vehicle.get_acceleration(vehicles[2], vehicles, c)
                      d1=Vehicle.get_acceleration(vehicles[3], vehicles, d)
                      Vehicle.update_state(vehicles[0],a1)
                      Vehicle.update_state(vehicles[1],b1)
                      Vehicle.update_state(vehicles[2],c1)
                      Vehicle.update_state(vehicles[3],d1)
                      for i in range(4):
                          reward_ = Vehicle.reward(vehicles[i], vehicles)
                          total_reward+=reward_
                          if total_reward > best_total_reward:
                              best_total_reward=total_reward
                              best_actions=[a1,b1,c1,d1]
                          else:
                              continue
                            # 恢复原始状态
                      for v, pos, vel in zip(vehicles, original_positions, original_velocities):
                          v.position = pos
                          v.velocity = vel    

    return best_actions, best_total_reward



# 定义车辆
# 创建车辆实例并设定路径点
vehicle_a = Vehicle(position=[0, 450], velocity=[3, 0], accelerate=[0, 0], target_position=[1000, 450], color=(255, 0, 0), name="A")
vehicle_b = Vehicle(position=[450, 0], velocity=[0, 3], accelerate=[0, 0], target_position=[450, 1000], color=(0, 255, 0), name="B")
vehicle_c = Vehicle(position=[1000,550], velocity=[-3,0], accelerate=[0, 0], target_position=[0,550], color=(0, 0, 255), name="C")
vehicle_d = Vehicle(position=[550,1000], velocity=[0, -3], accelerate=[0, 0], target_position=[550,0], color=(255, 255, 0), name="D")
vehicles=[vehicle_a,vehicle_b,vehicle_c,vehicle_d]
    
best_actions=predict_best_actions(vehicles, time_step=0.5)