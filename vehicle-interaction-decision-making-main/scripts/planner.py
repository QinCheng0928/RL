'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-05-13 23:24:21
LastEditTime: 2024-10-31 01:01:03
FilePath: /vehicle-interaction-decision-making/scripts/planner.py
Copyright 2024 puyu, All Rights Reserved.
'''

import math
import random
import logging
import numpy as np
from typing import Tuple, List

import utils
from utils import Node, StateList
from vehicle_base import VehicleBase

# 蒙特卡洛树搜索 (Monte Carlo Tree Search, MCTS)
# 流程：选择、扩展、模拟、反向传播
# 负责为自车选择最佳的控制动作
class MonteCarloTreeSearch:
    EXPLORATE_RATE = 1 / ( 2 * math.sqrt(2.0))
    LAMDA = 0.9
    WEIGHT_AVOID = 10
    WEIGHT_SAFE = 0.2
    WEIGHT_OFFROAD = 2
    WEIGHT_DIRECTION = 1
    WEIGHT_DISTANCE = 0.1
    WEIGHT_VELOCITY = 0.05

    def __init__(self, ego: VehicleBase, others: List[VehicleBase],
                 other_traj: List[StateList], cfg: dict = {}):
        self.ego_vehicle: VehicleBase = ego             # 当前需要规划行为的自车
        self.other_vehicle: VehicleBase = others        # 环境中其他车辆的状态
        self.other_predict_traj: StateList = other_traj # 其他车辆未来的运动轨迹

        self.computation_budget = cfg['computation_budget']
        self.dt = cfg['delta_t']


    # 接受一个根节点 (root) 并返回最优的子节点
    def excute(self, root: Node) -> Node:
        for _ in range(self.computation_budget):
            # 1. Find the best node to expand
            expand_node = self.tree_policy(root)
            # 2. Random run to add node and get reward
            reward = self.default_policy(expand_node)
            # 3. Update all passing nodes with reward
            self.update(expand_node, reward)

        return self.get_best_child(root, 0)

    # 该方法执行选择和扩展操作
    # 选择下一个节点(节点是通过选择动作进行扩展)
    def tree_policy(self, node: Node) -> Node:
        while node.is_terminal == False:
            if len(node.children) == 0:
                return self.expand(node)
            elif random.uniform(0, 1) < .5:
                node = self.get_best_child(node, MonteCarloTreeSearch.EXPLORATE_RATE)
            else:
                if node.is_fully_expanded == False:    
                    return self.expand(node)
                else:
                    node = self.get_best_child(node, MonteCarloTreeSearch.EXPLORATE_RATE)

        return node

    # 执行随机模拟，直到到达终止节点。
    # 返回终止时的奖励值
    def default_policy(self, node: Node) -> float:
        while node.is_terminal == False:
            cur_other_state = self.other_predict_traj[node.cur_level + 1]
            next_node = node.next_node(self.dt, cur_other_state)
            node = next_node

        return node.value

    # 更新通过的节点的访问次数和奖励值
    def update(self, node: Node, r: float) -> None:
        while node != None:
            node.visits += 1
            node.reward += r
            node = node.parent

    # 扩展当前节点
    # 它会尝试选择一个未被尝试过的动作，并根据动作生成一个新的子节点。
    def expand(self, node: Node) -> Node:
        tried_actions = [c.action for c in node.children]
        next_action = random.choice(utils.ActionList)
        while node.is_terminal == False and next_action in tried_actions:
            next_action = random.choice(utils.ActionList)
        cur_other_state = self.other_predict_traj[node.cur_level + 1]
        node.add_child(next_action, self.dt, cur_other_state)

        return node.children[-1]

    # 选择最佳子节点
    def get_best_child(self, node: Node, scalar: float) -> Node:
        bestscore = -math.inf
        bestchildren = []
        for child in node.children:
            exploit = child.reward / child.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / child.visits)
            score = exploit + scalar * explore
            if score == bestscore:
                bestchildren.append(child)
            if score > bestscore:
                bestchildren = [child]
                bestscore = score
        if len(bestchildren) == 0:
            logging.debug("No best child found, probably fatal !")
            return node

        return random.choice(bestchildren)

    # 计算当前节点的价值 
    # 在next_node add_child都会调用这个函数
    @staticmethod
    def calc_cur_value(node: Node, last_node_value: float) -> float:
        x, y, yaw = node.state.x, node.state.y, node.state.yaw
        step = node.cur_level
        ego_box2d = VehicleBase.get_box2d(node.state)
        ego_safezone = VehicleBase.get_safezone(node.state)

        avoid = 0
        safe = 0
        for cur_other_state in node.other_agent_state:
            if utils.has_overlap(ego_box2d, VehicleBase.get_box2d(cur_other_state)):
                avoid = -1
            if utils.has_overlap(ego_safezone, VehicleBase.get_safezone(cur_other_state)):
                safe = -1

        offroad = 0
        for rect in VehicleBase.env.rect:
            if utils.has_overlap(ego_box2d, rect):
                offroad = -1
                break

        direction = 0
        if MonteCarloTreeSearch.is_opposite_direction(node.state, ego_box2d):
            direction = -1

        delta_yaw = abs(yaw - node.goal_pos.yaw) % (np.pi * 2)
        delta_yaw = min(delta_yaw, np.pi * 2 - delta_yaw)
        distance = -(abs(x - node.goal_pos.x) + abs(y - node.goal_pos.y) + 1.5 * delta_yaw)

        cur_reward = MonteCarloTreeSearch.WEIGHT_AVOID * avoid + \
                     MonteCarloTreeSearch.WEIGHT_SAFE * safe + \
                     MonteCarloTreeSearch.WEIGHT_OFFROAD * offroad + \
                     MonteCarloTreeSearch.WEIGHT_DISTANCE * distance + \
                     MonteCarloTreeSearch.WEIGHT_DIRECTION * direction + \
                     MonteCarloTreeSearch.WEIGHT_VELOCITY * node.state.v

        total_reward = last_node_value + (MonteCarloTreeSearch.LAMDA ** (step - 1)) * cur_reward
        node.value = total_reward

        return total_reward

    # 检查是否在相反方向行驶
    @staticmethod
    def is_opposite_direction(pos: utils.State, ego_box2d = None) -> bool:
        x, y, yaw = pos.x, pos.y, pos.yaw
        if ego_box2d is None:
            ego_box2d = VehicleBase.get_box2d(pos)

        for laneline in VehicleBase.env.laneline:
            if utils.has_overlap(ego_box2d, laneline):
                return True

        lanewidth = VehicleBase.env.lanewidth

        # down lane
        if x > -lanewidth and x < 0 and (y < -lanewidth or y > lanewidth):
            if yaw > 0 and yaw < np.pi:
                return True
        # up lane
        elif x > 0 and x < lanewidth and (y < -lanewidth or y > lanewidth):
            if not (yaw > 0 and yaw < np.pi):
                return True
        # right lane
        elif y > -lanewidth and y < 0 and (x < -lanewidth or x > lanewidth):
            if yaw > 0.5 * np.pi and yaw < 1.5 * np.pi:
                return True
        # left lane
        elif y > 0 and y < lanewidth and (x < -lanewidth or x > lanewidth):
            if not (yaw > 0.5 * np.pi and yaw < 1.5 * np.pi):
                return True

        return False

    @staticmethod
    def initialize(cfg: dict = {}) -> None:
        MonteCarloTreeSearch.LAMDA = cfg['lamda']
        MonteCarloTreeSearch.WEIGHT_AVOID = cfg['weight_avoid']
        MonteCarloTreeSearch.WEIGHT_SAFE = cfg['weight_safe']
        MonteCarloTreeSearch.WEIGHT_OFFROAD = cfg['weight_offroad']
        MonteCarloTreeSearch.WEIGHT_DIRECTION = cfg['weight_direction']
        MonteCarloTreeSearch.WEIGHT_DISTANCE = cfg['weight_distance']
        MonteCarloTreeSearch.WEIGHT_VELOCITY = cfg['weight_velocity']


class KLevelPlanner:
    def __init__(self, cfg: dict = {}):
        # 表示自车将前瞻多少步进行规划
        self.steps = cfg['max_step']
        # 表示每步仿真的时间间隔
        self.dt = cfg['delta_t']
        # 包含配置信息的字典
        self.config = cfg

    # 返回自车的第一个控制动作和预测的轨迹
    def planning(self, ego: VehicleBase, others: List[VehicleBase]) -> Tuple[utils.Action, StateList]:
        other_prediction = self.get_prediction(ego, others)
        actions, traj = self.forward_simulate(ego, others, other_prediction)
        
        return actions[0], traj

    # 模拟自车的未来行为
    def forward_simulate(self, ego: VehicleBase, others: List[VehicleBase],
                         traj: List[StateList]) -> Tuple[List[utils.Action], StateList]:
        mcts = MonteCarloTreeSearch(ego, others, traj, self.config)
        current_node = Node(state = ego.state, goal = ego.target)
        current_node = mcts.excute(current_node)
        for _ in range(Node.MAX_LEVEL - 1):
            current_node = mcts.get_best_child(current_node, 0)

        actions = current_node.actions
        state_list = StateList()
        while current_node != None:
            state_list.append(current_node.state)
            current_node = current_node.parent
        expected_traj = state_list.reverse()

        if len(expected_traj) < self.steps + 1:
            logging.debug(f"The max level of the node is not enough({len(expected_traj)}),"
                          f"using the last value to complete it.")
            expected_traj.expand(self.steps + 1)

        # 返回ego的动作序列和expected_traj状态列表（轨迹）
        return actions, expected_traj

    def get_prediction(self, ego: VehicleBase, others: List[VehicleBase]) -> List[StateList]:
        pred_trajectory = []
        pred_trajectory_trans = []

        if ego.level == 0:
            for i in range(self.steps + 1):
                pred_traj: StateList = StateList()
                for other in others:
                    pred_traj.append(other.state)
                pred_trajectory.append(pred_traj)
            return pred_trajectory
        elif ego.level > 0:
            for idx in range(len(others)):
                # 检查每一辆其他车辆 others[idx] 是否已经到达目标（is_get_target）。
                # 如果该车辆已经到达目标，那么它的轨迹将直接沿用原有的状态，不再进行预测。
                # 否则，我们会对该车辆做进一步的交互预测。
                if others[idx].is_get_target:
                    pred_traj: StateList = StateList()
                    for i in range(self.steps + 1):
                        pred_traj.append(others[idx].state)
                    pred_trajectory_trans.append(pred_traj)
                    continue
                exchanged_ego: VehicleBase = others[idx]
                exchanged_ego.level = ego.level - 1
                exchanged_others: List[VehicleBase] = [ego]
                # 完善other列表
                for i in range(len(others)):
                    if i != idx:
                        exchanged_others.append(others[i])
                exchage_pred_others = self.get_prediction(exchanged_ego, exchanged_others)
                _, pred_idx_vechicle = self.forward_simulate(exchanged_ego, exchanged_others, exchage_pred_others)
                pred_trajectory_trans.append(pred_idx_vechicle)
        else:
            logging.error("get_prediction() excute error, the level must be >= 0 and > 3 !")
            return pred_trajectory

        for i in range(self.steps + 1):
            state = StateList()
            for states in pred_trajectory_trans:
                state.append(states[i])
            pred_trajectory.append(state)

        return pred_trajectory
