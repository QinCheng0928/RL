'''
Author: puyu <yuu.pu@foxmail.com>
Date: 2024-04-27 16:17:27
LastEditTime: 2024-10-31 01:01:31
FilePath: /vehicle-interaction-decision-making/scripts/vehicle_base.py
Copyright 2024 puyu, All Rights Reserved.
'''

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from utils import State
from env import EnvCrossroads


class VehicleBase(ABC):
    length = 5
    width = 2
    safe_length = 8
    safe_width = 2.4
    # Union[EnvCrossroads, None]
    # 意思是这个变量可以是一个 EnvCrossroads 类型的实例，也可以是 None，即没有值。
    env: Optional[EnvCrossroads] = None

    def __init__(self, name: str):
        self.name: str = name
        self.state: State = State()

    # 在给定位置创建一个给定朝向的车辆
    # 返回一个车辆的四个顶点的位置坐标列表
    @staticmethod
    def get_box2d(tar_offset: State) -> np.ndarray:
        # 定义车辆的矩形
        vehicle = np.array(
            [[-VehicleBase.length/2, VehicleBase.length/2,
              VehicleBase.length/2, -VehicleBase.length/2, -VehicleBase.length/2],
            [VehicleBase.width/2, VehicleBase.width/2,
             -VehicleBase.width/2, -VehicleBase.width/2, VehicleBase.width/2]]
        )
        # 定义旋转矩阵
        rot = np.array([[np.cos(tar_offset.yaw), -np.sin(tar_offset.yaw)],
                     [np.sin(tar_offset.yaw), np.cos(tar_offset.yaw)]])

        # 将车辆按照朝向旋转
        vehicle = np.dot(rot, vehicle)
        # 将车辆按照位置平移
        vehicle += np.array([[tar_offset.x], [tar_offset.y]])

        return vehicle

    # 原理同上，获取给定位置车辆的安全区域
    @staticmethod
    def get_safezone(tar_offset: State) -> np.ndarray:
        safezone = np.array(
            [[-VehicleBase.safe_length/2, VehicleBase.safe_length/2,
              VehicleBase.safe_length/2, -VehicleBase.safe_length/2, -VehicleBase.safe_length/2],
            [VehicleBase.safe_width/2, VehicleBase.safe_width/2,
             -VehicleBase.safe_width/2, -VehicleBase.safe_width/2, VehicleBase.safe_width/2]]
        )
        rot = np.array([[np.cos(tar_offset.yaw), -np.sin(tar_offset.yaw)],
                     [np.sin(tar_offset.yaw), np.cos(tar_offset.yaw)]])

        safezone = np.dot(rot, safezone)
        safezone += np.array([[tar_offset.x], [tar_offset.y]])

        return safezone

    # 初始化
    @staticmethod
    def initialize(env: EnvCrossroads, len: float, width: float,
                   safe_len: float, safe_width: float):
        VehicleBase.env = env
        VehicleBase.length = len
        VehicleBase.width = width
        VehicleBase.safe_length = safe_len
        VehicleBase.safe_width = safe_width
