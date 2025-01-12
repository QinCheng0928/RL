'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-05-28 23:07:35
LastEditTime: 2024-10-31 01:01:18
FilePath: /vehicle-interaction-decision-making/scripts/run.py
Copyright 2024 puyu, All Rights Reserved.
'''

import os
import time
import yaml
import logging
import argparse
from datetime import datetime
from typing import List
from concurrent.futures import ProcessPoolExecutor, Future

import matplotlib.pyplot as plt
import gym
import torch
from ppo.ppo_agent import PPOAgent

from utils import Node, ActionList, State, kinematic_propagate
from env import EnvCrossroads
from vehicle_base import VehicleBase
from vehicle import Vehicle, VehicleList
from planner import MonteCarloTreeSearch
import itertools

LOG_LEVEL_DICT = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING,
                  "error": logging.ERROR, "critical": logging.CRITICAL}

# Actor输出值与n个车k值的对应关系
# 返回Actor索引对应车辆的k值
def get_kvalue(num_agents, num_kvalue, maximum_probability_index, vehicle_index):
    permutations = itertools.product(range(num_kvalue), repeat=num_agents)
    # itertools.product 会生成长度为 num_agents 的所有组合，每个元素从 0 到 num_kvalue-1 中选取。

    return permutations[maximum_probability_index][vehicle_index]


# Critic输出值与n个车动作的对应关系
# 返回Critic索引对应车辆的动作
def get_action(num_agents, maximum_value_index, vehicle_index):
    permutations = itertools.product(range(6), repeat=num_agents)
    
    return ActionList[permutations[maximum_value_index][vehicle_index]]



def run(rounds_num:int, config_path:str, save_path:str, no_animation:bool, save_fig:bool) -> None:
    # 读取配置文件并加载到 config 字典中
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        logging.info(f"Config parameters:\n{config}")

    # 初始化环境和车辆
    logging.info(f"rounds_num: {rounds_num}")
    map_size = config["map_size"]
    lane_width = config["lane_width"]
    env = EnvCrossroads(map_size, lane_width)
    delta_t = config['delta_t']
    max_simulation_time = config['max_simulation_time']
    vehicle_draw_style = config['vehicle_display_style']
    max_episodes = config["max_episodes"]
    max_timesteps = config["max_timesteps"]
    update_timestep = config["update_timestep"]
    print_freq = config["print_freq"]
    
    # initialize
    VehicleBase.initialize(env, 5, 2, 8, 2.4)
    MonteCarloTreeSearch.initialize(config)
    Node.initialize(config['max_step'], MonteCarloTreeSearch.calc_cur_value)
    
    vehicles = VehicleList()
    for vehicle_name in config["vehicle_list"]:
        vehicle = Vehicle(vehicle_name, config)
        vehicles.append(vehicle)

    # PPO 相关初始化
    state = State()
    state_dim = state.get_num_attributes()      # 状态维度为状态列表的属性个数
    action_dim = len(ActionList)                # 动作维度为动作列表的长度
    num_agents = len(vehicles)                  # 智能体车辆的数量
    num_kvalue = 4                              # k 取值的个数
    print(f"num_agents: {num_agents}")
    ppo_agent = PPOAgent(state_dim, action_dim, num_agents, lr=0.002, gamma=0.99, K_epochs=4, eps_clip=0.2)

    


   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--rounds', '-r', type=int, default=5, help='')
    parser.add_argument('--output_path', '-o', type=str, default=None, help='')
    parser.add_argument('--log_level', '-l', type=str, default="info",
                        help=f"debug:logging.DEBUG\tinfo:logging.INFO\t"
                             f"warning:logging.WARNING\terror:logging.ERROR\t"
                             f"critical:logging.CRITICAL\t")
    parser.add_argument('--config', '-c', type=str, default=None, help='')
    parser.add_argument('--no_animation', action='store_true', default=False, help='')
    parser.add_argument('--save_fig', action='store_true', default=False, help='')
    args = parser.parse_args()

    current_file_path = os.path.abspath(__file__)
    # os.path.dirname(path)：
    # 这个函数返回去掉路径中的最后一个组件文件或文件夹的部分。
    if args.output_path is None:
        # 退回vehicle-interaction-decision-making-main文件夹
        args.output_path = os.path.dirname(os.path.dirname(current_file_path))
    if args.config is None:
        config_file_path = os.path.join(os.path.dirname(os.path.dirname(current_file_path)),
                                        'config', 'unprotected_left_turn.yaml')
    else:
        config_file_path = args.config

    log_level = LOG_LEVEL_DICT[args.log_level]
    log_format = '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    result_save_path = os.path.join(args.output_path, "logs", formatted_time)
    if args.save_fig:
        os.makedirs(result_save_path, exist_ok=True)
        log_file_path = os.path.join(result_save_path, 'log')
        logging.basicConfig(level=log_level, format=log_format,
                            handlers=[logging.StreamHandler(), logging.FileHandler(filename=log_file_path)])
        logging.info(f"Experiment results save at \"{result_save_path}\"")
    else:
        logging.basicConfig(level=log_level, format=log_format, handlers=[logging.StreamHandler()])
    logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
    logging.info(f"log level : {args.log_level}")

    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    run(args.rounds, config_file_path, result_save_path, args.no_animation, args.save_fig)
