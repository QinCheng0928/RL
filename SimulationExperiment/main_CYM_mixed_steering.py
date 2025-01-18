# source ../../install/setup.bash
import carla
import rclpy
from rclpy.node import Node
import sys
import traceback
import math
import time
from MultiProxy import Timeout_Judge

from carla_msgs.msg import CarlaEgoVehicleStatus,CarlaEgoVehicleControl,CarlaEgoVehicleInfo,CarlaActorList
from derived_object_msgs.msg import ObjectArray
from nav_msgs.msg import Odometry
import os
print(os.getcwd())
from cq.agents.navigation.global_route_planner import GlobalRoutePlanner
from cq.agents.navigation.controller import VehiclePIDController
from cq.agents.tools.misc import draw_waypoints, cal_distance, cal_distance2, vector, get_speed, cal_angle2
from cq_utils import quaternion_to_euler_object, status_process, Odometry_process, object_process, clock_process
import numpy as np
import asyncio

import pickle
def save_varible(v,filename):
    f = open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variable(filename):
    f=open(filename,'rb+')
    r=pickle.load(f)
    f.close()
    return r

global DATA_Real
global DATA_Cal

global current_time

# 获取当前时间并格式化为字符串
current_time = time.strftime("%Y-%m%d-%H-%M-%S")

# 创建新的文件夹，文件夹名包含当前时间
folder_name = f"Case_3CAV1HV_{current_time}"
os.makedirs('./Case_3CAV1HV/'+'Case_3CAV1HV_{}'.format(current_time), exist_ok=True)

DATA_REAL = {
    'hero1':{'xy':[], 'x':[], 'v':[], 'a':[], 'yaw':[]}, 
    'hero2':{'xy':[], 'x':[], 'v':[], 'a':[], 'yaw':[]},
    'hero3':{'xy':[], 'x':[], 'v':[], 'a':[], 'yaw':[]},
    'hero':{'xy':[], 'x':[], 'v':[], 'a':[], 'yaw':[]}
    }

DATA_CAL = {
    'hero1':{'xy':[], 'x':[], 'v':[], 'a':[], 'yaw':[]}, 
    'hero2':{'xy':[], 'x':[], 'v':[], 'a':[], 'yaw':[]},
    'hero3':{'xy':[], 'x':[], 'v':[], 'a':[], 'yaw':[]},
    'hero':{'xy':[], 'x':[], 'v':[], 'a':[], 'yaw':[]}
    }

TIME_RECORD = {
    'time':[]
    }

DATA_THETA = {'hero':{'ALL_n_theta':[],
                      '_t_Theta':[]}}

save_varible(DATA_REAL,'./Case_3CAV1HV/'+'Case_3CAV1HV_{}'.format(current_time) + '/DATA_REAL.txt')
save_varible(DATA_CAL,'./Case_3CAV1HV/'+'Case_3CAV1HV_{}'.format(current_time) + '/DATA_CAL.txt')
save_varible(DATA_THETA,'./Case_3CAV1HV/'+'Case_3CAV1HV_{}'.format(current_time) + '/DATA_THETA.txt')
save_varible(TIME_RECORD,'./Case_3CAV1HV/'+'Case_3CAV1HV_{}'.format(current_time) + '/TIME_RECORD.txt')

    
# PID
#args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': 0.05}
# args_lateral_dict = {'K_P': -1.95,'K_D': -0.2,'K_I': -0.07,'dt': 0.1}
args_lateral_dict = {'K_P': -0.65,'K_D': -0.2,'K_I': -5.2,'dt': 0.01}


#args_long_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': 0.05}
args_long_dict = {'K_P': 1.0,'K_D': 0.0,'K_I': 0.0,'dt': 0.01}


client = carla.Client('localhost',2000)

world = client.get_world()

# Load layered map for Town 03 with minimum layout plus buildings and parked vehicles
# world = client.load_world('tongji', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
# Toggle all buildings off 关闭显示所有建筑物
# world.unload_map_layer(carla.MapLayer.Buildings)

# # Toggle all buildings on   打开显示所有建筑物
# world.load_map_layer(carla.MapLayer.Buildings)


m = world.get_map()
#设置视角
transform = carla.Transform()
spectator = world.get_spectator()
bv_transform = carla.Transform(carla.Location(z=100,x=-616.4,y=71.4), carla.Rotation(yaw=0, pitch=-90))
spectator.set_transform(bv_transform)

# blueprint_library = world.get_blueprint_library()
spawn_points = m.get_spawn_points()
# a = spawn_points[1].get_forward_vector()

# for i, spawn_point in enumerate(spawn_points):
#     world.debug.draw_string(spawn_point.location, str(i), life_time=100)
#     world.debug.draw_arrow(spawn_point.location, spawn_point.location + spawn_point.get_forward_vector(), life_time=100)

# 交叉口可规划轨迹：22->872,787->368,871->368

# global path planner
distance = 2.0
global_path_idx = {'hero1':[22,872], 'hero2':[787,368], 'hero3':[642,377]}
grp = GlobalRoutePlanner(m, distance)
global_path = {'hero1':[], 'hero2':[], 'hero3':[]}
T=10000
for av_rolename in global_path_idx:
    # print(spawn_points[global_path_idx[av_rolename][0]])
    origin = spawn_points[global_path_idx[av_rolename][0]].location
    destination = spawn_points[global_path_idx[av_rolename][1]].location
    route = grp.trace_route(origin, destination)
    # for i in route:
    #     print(av_rolename,i[0].transform.location.x,-i[0].transform.location.y)
    for pi, pj in zip(route[:-1], route[1:]):
        pi_location = pi[0].transform.location
        pj_location = pj[0].transform.location 
        pi_location.z = 0.5
        pj_location.z = 0.5
        world.debug.draw_line(pi_location, pj_location, thickness=0.2, life_time=T, color=carla.Color(b=255))
        pi_location.z = 0.6
        world.debug.draw_point(pi_location, color=carla.Color(b=255), life_time=T)


    #waypoint和ros中是沿着y轴反转的，y和yaw都是相反数，但是way_point.transform不允许修改，只能在后面控制和计算距离的时候修改
    global_path[av_rolename] = [way_point[0]for way_point in route]



'''Game初始化的相关内容 & Game与Carla连接的相关内容'''
# Carla_Role 与 Game_Veh 的对应关系
from Vehicle import CAVandHV_allist
from param import *

# 初始化Game_Vehicle
S_all, S_cav, S_hv= CAVandHV_allist( Road_id=1, Sceanrio_id=5, Vehcom_id=1, Case_id=1)

# 车辆状态初始赋值
for i in range(len(S_all)):
    S_all[i].Attribute()
    S_all[i].X = S_all[i].init_x
    # print("车辆{}的纵向位置".format(i),S_all[i].init_x)
    S_all[i].V = S_all[i].init_v
    # print("车辆{}的速度".format(i),S_all[i].init_v)
    S_all[i].A = S_all[i].init_a
    # print("车辆{}的加速度".format(i),S_all[i].init_a)
    S_all[i].Anew_real = S_all[i].init_a
    S_all[i].Anew_infer = S_all[i].init_a
 
# 初始化Carla_roles
from data_transf import Role
Role_all = []
Role_AV = []
Role_HV = []
for rolename in Role_direction.keys():
    role = Role(rolename= rolename)
    Role_all.append(role)
    if role.rolename in Rolename_AV:
        Role_AV.append(role)
    else:
        Role_HV.append(role)
# print('Role_AV', Role_AV)
# print('Role_HV', Role_HV)

# Game车辆与Carla角色的映射
def Carla_Game_map(S_all, Role_all):
    Gveh_Crole = {}
    Crole_Gveh = {}
    for veh in S_all:
        for role in Role_all:
            if Role_direction[role.rolename] == veh.info1:
                Gveh_Crole[veh] = role
                Crole_Gveh[role] = veh
    return Gveh_Crole, Crole_Gveh

Gveh_Crole, Crole_Gveh = Carla_Game_map(S_all, Role_all)
# Gveh_Crole——将Game中车辆对应到Carla中
# Crole_Gveh——将Carla中车辆对应到Game中

# 仿真参数设置
delta_t = 0.2

# 构建一个Game实例
from DPUGame import DPUGAME
from data_transf import CONF_MATRIXALL
Conf_indexall = CONF_MATRIXALL

GAME = DPUGAME(S_all, Conf_indexall=Conf_indexall, current_timestamps=0, delta_t = delta_t)
print(GAME)


''''''
# /carla/objects内有所有车辆信息，包括AV、HV、SV，但无法获得控制量
# hero?为多辆AV，带hero的topic都为自车的信息（除了carla/hero?/objects为除了自车之外的他车信息），/carla/hero1/odometry中有位置、朝向、速度，/carla/hero1/vehicle_control_cmd中有油门刹车方向盘
# manual_control.py启动键盘控制，默认会启动/carla/hero/...，其中/carla/hero/vehicle_status为键盘控制的车的速度(标量)、三轴加速度和角加速度、油门刹车方向盘！！！\
# 注意manual_control.py的role_name参数默认为hero不需要修改，只有名字为hero/hero0-10/ego_vehicle才会有/carla/hero/vehicle_status的topic，创建AV的objects.json也是这样
# 注：objects、status、Odometry中的四元数都是一样的, 计算出来的yaw就是沿x轴逆时针的角度（弧度制）,和heading应该是一个意思
class sub_pub(Node):
    def __init__(self):
        super().__init__('listener')

        # self.all_veh_info = {'AV':{},'HV':{'ego_vehicle':{}},'SV':{}}
        self.AV_Odometry = {}
        self.veh_status = {}
        self.rolename_vehid_dict = {'AV':{},'HV':{},'SV':{}}
        self.vehid_object_dict = {}

        self.idx_dict = {"hero1":0,"hero2":0,"hero3":0,"hero4":0}#路径跟踪的索引点
        
        # 在/carla/hero1/vehicle_control_cmd中，可以订阅所有车的id
        self.sub_actor_list = self.create_subscription(CarlaActorList, '/carla/actor_list', self.ck_actor_list, 10)
        # 在/carla/objects中，可以订阅所有车的三轴位置和朝向、三轴速度和角速度，只有AV（和HV）可以获得方向盘转角
        self.sub_objects = self.create_subscription(ObjectArray, '/carla/objects', self.ck_objects, 10)

        # 控制需要前一时刻的控制量,可以订阅AV和HV的当前速度(标量)、加速度、方向、油门刹车方向盘
        # AV
        self.sub_status_hero1 = self.create_subscription(CarlaEgoVehicleStatus, '/carla/hero1/vehicle_status', self.ck_status_hero1, 10)
        self.sub_status_hero2 = self.create_subscription(CarlaEgoVehicleStatus, '/carla/hero2/vehicle_status', self.ck_status_hero2, 10) 
        self.sub_status_hero3 = self.create_subscription(CarlaEgoVehicleStatus, '/carla/hero3/vehicle_status', self.ck_status_hero3, 10)
        self.sub_status_hero4 = self.create_subscription(CarlaEgoVehicleStatus, '/carla/hero4/vehicle_status', self.ck_status_hero4, 10)
        # HV
        self.sub_status_hero = self.create_subscription(CarlaEgoVehicleStatus, '/carla/hero/vehicle_status', self.ck_status_hero, 10)
        
        # 在/carla/hero1/odometry中，可以订阅AV的三轴位置和朝向、三轴速度和角速度
        self.sub_Odometry_hero1 = self.create_subscription(Odometry, '/carla/hero1/odometry', self.ck_odometry_hero1, 10)
        self.sub_Odometry_hero2 = self.create_subscription(Odometry, '/carla/hero2/odometry', self.ck_odometry_hero2, 10)
        self.sub_Odometry_hero3 = self.create_subscription(Odometry, '/carla/hero3/odometry', self.ck_odometry_hero3, 10)
        self.sub_Odometry_hero4 = self.create_subscription(Odometry, '/carla/hero4/odometry', self.ck_odometry_hero4, 10)
        
        # 在/carla/hero1/vehicle_control_cmd中，可以发布油门、刹车、方向盘信息对AV进行控制（ros生成的AV默认是自动驾驶模式不需修改，手动模式的topic为/carla/hero1/vehicle_control_cmd_manual）
        self.pub_control_cmd_hero1 = self.create_publisher(CarlaEgoVehicleControl, '/carla/hero1/vehicle_control_cmd', 10)
        self.pub_control_cmd_hero2 = self.create_publisher(CarlaEgoVehicleControl, '/carla/hero2/vehicle_control_cmd', 10)
        self.pub_control_cmd_hero3 = self.create_publisher(CarlaEgoVehicleControl, '/carla/hero3/vehicle_control_cmd', 10)
        self.pub_control_cmd_hero4 = self.create_publisher(CarlaEgoVehicleControl, '/carla/hero4/vehicle_control_cmd', 10)
        
        # 在/carla/clock中，可以订阅当前仿真时间
        self.sub_SimulationTime = self.create_subscription(CarlaEgoVehicleStatus, '/carla/hero1/vehicle_status', self.ck_stamps, 10)

        # 保存最后一次计算结果
        self.RESULT = None

        self.DATA_REAL = DATA_REAL
        self.DATA_CAL = DATA_CAL
        self.DATA_THETA = DATA_THETA
        self.TIME_RECORD = TIME_RECORD

        self.Game = GAME

    # 更新所有车，建立AV的role_name与id的映射关系
    def ck_actor_list(self,data):
        # actor_list -> Dict{hero_i:object_id}
        self.rolename_vehid_dict = {'AV':{},'HV':[],'SV':[]}
        for act in data.actors:
            if act.rolename == 'hero' or act.rolename == 'default':#hero为键盘控制车的默认名字（可修改），为方向盘控制车的默认名字（不知道能不能改成有rostopic的名称）
                self.rolename_vehid_dict['HV'].append(act.parent_id)
            elif act.rolename.startswith('hero'):
                self.rolename_vehid_dict['AV'][act.rolename] = act.id
            elif act.rolename == 'sumo_driver':
                self.rolename_vehid_dict['SV'].append(act.id)
        # print(self.rolename_vehid_dict)

    # 更新所有车的三轴位置和朝向、三轴速度和角速度
    def ck_objects(self,data):
        # print(data[0],data[1])
        self.vehid_object_dict = {}
        for object in data.objects:
            self.vehid_object_dict[object.id] = object_process(object) # DO: 加速度二维计算
    
    # 获取当前仿真时间
    def ck_stamps(self,data):
        self.time_stampssec = clock_process(data)

    # AV
    def ck_status_hero1(self,data):
        self.veh_status['hero1'] = status_process(data)
        # print(self.veh_status)
    def ck_status_hero2(self,data):
        self.veh_status['hero2'] = status_process(data)
    def ck_status_hero3(self,data):
        self.veh_status['hero3'] = status_process(data)
    def ck_status_hero4(self,data):
        self.veh_status['hero4'] = status_process(data)
    # HV
    def ck_status_hero(self,data):
        self.veh_status['hero'] = status_process(data)
    
    # AV
    def ck_odometry_hero1(self,data):
        self.AV_Odometry['hero1'] = Odometry_process(data)
        self.get_logger().info('I heard: "%s"' % Odometry_process(data))
    def ck_odometry_hero2(self,data):
        self.AV_Odometry['hero2'] = Odometry_process(data)
    def ck_odometry_hero3(self,data):
        self.AV_Odometry['hero3'] = Odometry_process(data)
    def ck_odometry_hero4(self,data):
        self.AV_Odometry['hero4'] = Odometry_process(data)

    def compute(self):
        # self.actor_list + self.objects + self.veh_status -> path[其实next_point足够，但是可能过点需要用下个点],target_speed
        #-> Dict{'hero_':(past_steering,current_speed,get_transform)}
        # print(self.Game)
        # print('self.AV_Odometry', self.AV_Odometry)
        try:
            # AV状态获取
            # print('self.AV_Odometry', self.AV_Odometry)
            for av in Role_AV:
                rolename = av.rolename
                # print(rolename, self.AV_Odometry[rolename])
                xx, xy = self.AV_Odometry[rolename]['loc']
                v = self.veh_status[rolename]['v']
                a = self.veh_status[rolename]['a']  # DONE: 加速度计算
                yaw = self.veh_status[rolename]['yaw']
                
                Crole_Gveh[av].V = min(v, Crole_Gveh[av].max_velocity)
                Crole_Gveh[av].A = np.clip(a, Crole_Gveh[av].max_accl, Crole_Gveh[av].max_decel) 
                Crole_Gveh[av].X = av.get_1D_x((xx, xy))
                av.get_next_npoint_mixed(5, (xx, xy)) # 获取下一时刻的4个参考点位置（速度计算方法？？？
                # print(rolename, (xx, xy))

                self.DATA_REAL[rolename]['xy'].append((xx, xy))
                self.DATA_REAL[rolename]['x'].append(Crole_Gveh[av].X)
                self.DATA_REAL[rolename]['v'].append(v)
                self.DATA_REAL[rolename]['a'].append(a)
                self.DATA_REAL[rolename]['yaw'].append(yaw)
            # print(self.DATA_REAL)
            
            # print("---------------------CAV状态获取成功----------------------")

            # HV状态获取
            # print('HV的rolename',self.rolename_vehid_dict['HV'], self.vehid_object_dict)
            for HV_id in self.rolename_vehid_dict['HV']: #只有一辆
                if HV_id in self.vehid_object_dict.keys():
                    # print('HV收集动作', self.vehid_object_dict)
                    HV_x, HV_y, HV_yaw, HV_v, HV_a = self.vehid_object_dict[HV_id]  # DONE: 加速度计算
                    # print('HV_x, HV_y, HV_yaw, HV_v', HV_x, HV_y, HV_yaw, HV_v)
                    hv = Role_HV[0]
                    rolename = hv.rolename
                    
                    xx, xy = HV_x, HV_y
                    v, a = HV_v, HV_a

                    Crole_Gveh[hv].V = min(v, Crole_Gveh[hv].max_velocity)
                    Crole_Gveh[hv].A = np.clip(a, Crole_Gveh[hv].max_accl, Crole_Gveh[hv].max_decel) 
                    Crole_Gveh[hv].X = hv.get_1D_x((xx, xy))

                    self.DATA_REAL[rolename]['xy'].append((xx, xy))
                    self.DATA_REAL[rolename]['x'].append(Crole_Gveh[av].X)
                    self.DATA_REAL[rolename]['v'].append(v)
                    self.DATA_REAL[rolename]['a'].append(a)
                    self.DATA_REAL[rolename]['yaw'].append(HV_yaw)
            # print("---------------------HV状态获取成功----------------------")
            
            self.TIME_RECORD['time'].append(self.time_stampssec)
            print('当前仿真时间', self.time_stampssec)

            # AV动作计算
            try:
                time1 = time.time()
                for av in Role_AV:
                    if len(av.future_points) != 0:
                        Crole_Gveh[av].precision = np.linalg.norm(np.array(av.future_points[0]) - np.array(self.AV_Odometry[av.rolename]['loc']))
                    else:
                        Crole_Gveh[av].precision = 0
                for hv in Role_HV:
                    Crole_Gveh[hv].precision = 3
                # time.sleep(0.3)
                # 匀速行驶
                # Sall_A, Sall_V = ([1 for i in range(len(self.Game.S_all))], [4, 0, 0, 10])#5-i*0.5 for i in range(len(self.Game.S_all))

                # 无超时判断
                trigger = fun(S_all)
                if not trigger:
                    Sall_A, Sall_V = self.IDM.update()  # TODO
                else:
                    Sall_A, Sall_V = self.Game.update()  # TODO

                # Timeout_Judge(time_limit=0.3, Game_class=self.Game)
                # if self.Game._cal_stage==0:
                #     self.Game = self.Game.Now_Game_Init
                #     Sall_A, Sall_V = [self.Game.S_all[i].Anew_real for i in range(len(self.Game.S_all))], [min(np.sqrt((self.Game.S_all[i].V)**2 + 2* self.Game.S_all[i].Anew_real*self.Game.S_all[i].precision), self.Game.S_all[i].max_velocity) for i in range(len(self.Game.S_all))]

                # elif self.Game._cal_stage==1:
                #     self.Game = self.Game.Now_Game_bestCal
                #     Sall_A, Sall_V = [self.Game.S_all[i].Anew_real for i in range(len(self.Game.S_all))], [min(np.sqrt((self.Game.S_all[i].V)**2 + 2* self.Game.S_all[i].Anew_real*self.Game.S_all[i].precision), self.Game.S_all[i].max_velocity) for i in range(len(self.Game.S_all))]

                # elif self.Game._cal_stage==2:
                #     self.Game = self.Game
                #     Sall_A, Sall_V = [self.Game.S_all[i].Anew_real for i in range(len(self.Game.S_all))], [min(np.sqrt((self.Game.S_all[i].V)**2 + 2* self.Game.S_all[i].Anew_real*self.Game.S_all[i].precision), self.Game.S_all[i].max_velocity) for i in range(len(self.Game.S_all))]

                time2 = time.time()
                # 计算耗时
                elapsed_time = time2 - time1

                # 显示耗时
                print(f"程序执行耗时：{elapsed_time} 秒")
                print('Sall_A', Sall_A)
                print('Sall_V', Sall_V)
            except Exception as e:
                print('计算步骤出现错误!!')
                time.sleep(0.3)
                print(e)
                print(sys.exc_info())

                print('\n', '>>>' * 20)
                int(traceback.print_exc())

            print("---------------------Game计算成功----------------------")
            result = {"hero1":{'planning_path':1,'target_speed':2},
                        "hero2":{'planning_path':1,'target_speed':2},
                        "hero3":{'planning_path':1,'target_speed':2},
                        "hero4":{'planning_path':1,'target_speed':2}}
            
            for hv_role in Role_HV:
                rolename = hv_role.rolename
                self.DATA_CAL[rolename]['v'].append(Sall_V[self.Game.S_all.index(Crole_Gveh[hv_role])])
                self.DATA_CAL[rolename]['a'].append(Sall_A[self.Game.S_all.index(Crole_Gveh[hv_role])])
                self.DATA_THETA[rolename]['ALL_n_theta'] = Crole_Gveh[hv_role].ALL_n_theta
                self.DATA_THETA[rolename]['_t_Theta'] = Crole_Gveh[hv_role]._t_Theta
            for rolename in result.keys():
                for role in Role_AV:
                    if role.rolename == rolename:
                        result[role.rolename]['target_speed'] = Sall_V[self.Game.S_all.index(Crole_Gveh[role])]  # TODO
                        if math.isnan(Sall_V[self.Game.S_all.index(Crole_Gveh[role])]):
                            result[role.rolename]['target_speed'] = 0
                        result[role.rolename]['planning_path'] = role.future_points

                        self.DATA_CAL[rolename]['v'].append(Sall_V[self.Game.S_all.index(Crole_Gveh[role])])
                        self.DATA_CAL[rolename]['a'].append(Sall_A[self.Game.S_all.index(Crole_Gveh[role])])
                        
                        break
            self.RESULT = result
            # print("compute success!")
            
        except Exception as e:
            result = None
            print(e)
            print(sys.exc_info())

            print('\n', '>>>' * 20)
            print(traceback.print_exc())
        return result    
        
    def av_control(self, result):
        # print(self.rolename_vehid_dict)
        
        # for HV_id in self.rolename_vehid_dict['HV']: #只有一辆
        #     print(self.vehid_object_dict)
        #     if HV_id in self.vehid_object_dict.keys():
        #         print()
        #         HV_x, HV_y, HV_yaw, HV_v = self.vehid_object_dict[HV_id]
                

                
        # print(self.vehid_object_dict)
        # print('HV',HV_id,HV_x, HV_y, HV_yaw, HV_v)#速度和角度可能得看下，compass毕竟不一样，但应该没错
        # self.vehid_object_dict[id]  #HV和虚拟车需要去/carla/objects中通过id获得位置信息（虚拟车必须以此方式获得速度，而HV还可以从statas中拿到）
        for SV_id in self.rolename_vehid_dict['SV']:
            try:
                SV_x, SV_y, SV_yaw, SV_v = self.vehid_object_dict[SV_id]
                # print('SV', SV_id, SV_x, SV_y, SV_yaw, SV_v)
            except:
                pass
        

        control_result = {}
        # 修改:实时输入和控制（可以实时调整PID参数，虽然可能没啥用
        for rolename in result:
            # print('-----------------',rolename,'-----------------')
            if rolename not in self.rolename_vehid_dict['AV']:
                break
            if rolename not in self.AV_Odometry:
                break
            if rolename not in self.veh_status:
                break
            
            
            get_transform = carla.libcarla.Transform()
            get_transform.location.x, get_transform.location.y  = self.AV_Odometry[rolename]['loc']
            get_transform.rotation.yaw = self.AV_Odometry[rolename]['Odometry_yaw']

            veh_dict = {"past_steering":self.veh_status[rolename]['control_value'][2],
                        "current_speed":self.veh_status[rolename]['v'],
                        "get_transform":get_transform}
            PID=VehiclePIDController(veh_dict,args_lateral=args_lateral_dict,args_longitudinal=args_long_dict)

            #根据速度与跟踪点进行控制
            target_speed = 0

            next_list = global_path[rolename]
            
            '''求解车辆在下一时刻的参考点，定义并输入预期属性值'''
            def next_point_transformtype():
                # 临时
                # point1 = (next_list[self.idx_dict[rolename]].transform.location.x, next_list[self.idx_dict[rolename]].transform.location.y)
                # point2 = (next_list[self.idx_dict[rolename]].transform.location.x + 2, next_list[self.idx_dict[rolename]].transform.location.y + 2)


                nextpoint_indx = self.idx_dict[rolename] #选取的参考点的索引
                # point1 = (get_transform.location.x, get_transform.location.y)#result[rolename]['planning_path'][nextpoint_indx]
                next = carla.libcarla.Transform()

                if len(result[rolename]['planning_path'])==0:
                    next.location.x, next.location.y = get_transform.location.x+0.1, get_transform.location.y+0.1
                    next.rotation.yaw = get_transform.rotation.yaw
                else:
                    point2 = result[rolename]['planning_path'][0] # result[rolename]['planning_path'][nextpoint_indx + 1]
                    next.location.x, next.location.y = point2[0], point2[1]#result[rolename]['planning_path'][nextpoint_indx]
                    # print('当前轨迹点和追踪轨迹点', point1, (next.location.x, next.location.y))
                    def get_yaw(point1, point2):
                        delta_x = point2[0] - point1[0]
                        delta_y = point2[1] - point1[1]
                        yaw = np.arctan2(delta_y, delta_x)
                        return yaw # 弧度
                    
                    if len(result[rolename]['planning_path'])>2:
                        #航向角计算
                        point1 = result[rolename]['planning_path'][1] 
                        next.rotation.yaw = np.degrees(get_yaw(point2,point1))
                        # next.rotation.yaw = np.degrees(get_yaw(point1, (next.location.x, next.location.y)))#get_yaw(result[rolename]['planning_path'][nextpoint_indx], result[rolename]['planning_path'][nextpoint_indx + 1])
                        # print('当前航向角和控制的航向角', get_transform.rotation.yaw, next.rotation.yaw)
                    else:
                        next.rotation.yaw = get_transform.rotation.yaw
                # spectator.set_transform(carla.Transform(next.location))
                return next
            ''''''
            '''注释掉'''
            # next.transform.location.x = 0
            # next.transform.location.x, next.transform.location.y = 0, 0#next1.transform.x, next1.transform.y
            # next.transform.rotation.pitch, next.transform.rotation.yaw, next.transform.rotation.roll = 0,0,0

            # spectator.set_transform(carla.Transform(next.transform.location))
            '''注释掉'''
            next = next_point_transformtype()
            track_dist = cal_distance2(get_transform, next)# 注意next的y和yaw是相反数，不过cal_distance2和PID.run_step的计算过程都是正确的
            track_angle = cal_angle2(get_transform, next)
            # print('len(next_list)', len(next_list))

            # if self.idx_dict[rolename] == len(next_list) - 1:
            #     target_speed = 0
            if track_dist < 4 or abs(track_angle) <180:
                self.idx_dict[rolename] += 1
                # print(self.idx_dict[rolename],len(next_list))
                # next = next_list[self.idx_dict[rolename]]
                target_speed = result[rolename]['target_speed']
                if len(result[rolename]['planning_path']) <= 4:
                    target_speed = 0
                next = next_point_transformtype()
                track_dist = cal_distance2(get_transform, next)# 注意next的y和yaw是相反数，不过cal_distance2和PID.run_step的计算过程都是正确的
                # track_angle = cal_angle2(get_transform, next)
            else:
                print('超出控制能力，速度降为00000！')
                print(self.idx_dict[rolename],get_transform,next, track_dist,track_angle)
                # sys.exit()
                pass
            # print('PID目标速度',target_speed)
            # print('PID控制目标', next)
            control = PID.run_step(target_speed, next, y_filp=True)
            # if rolename=='hero3':
            #     
            real_control = CarlaEgoVehicleControl()
            real_control.throttle = control.throttle
            real_control.steer = control.steer
            real_control.brake = control.brake
            control_result[rolename] = real_control
            # print(type(control))
        try:
            self.pub_control_cmd_hero1.publish(control_result['hero1'])
        except:
            pass
        try:
            self.pub_control_cmd_hero2.publish(control_result['hero2'])
        except:
            pass
        try:
            self.pub_control_cmd_hero3.publish(control_result['hero3'])
        except:
            pass
        try:
            self.pub_control_cmd_hero4.publish(control_result['hero4'])
        except:
            pass
    
    async def task1(self):
        while True:
            self.DATA_REAL = load_variable('./Case_3CAV1HV/'+'Case_3CAV1HV_{}'.format(current_time) + '/DATA_REAL.txt')
            self.DATA_CAL = load_variable('./Case_3CAV1HV/'+'Case_3CAV1HV_{}'.format(current_time) + '/DATA_CAL.txt')
            time_1 = time.time()
            self.compute()
            time_2 = time.time()
            print('整体计算COMPUTE模块耗时{}'.format(time_2 - time_1))
            rclpy.spin_once(self)
            # await result_queue.put(self.result)
            save_varible(self.DATA_REAL, './Case_3CAV1HV/'+'Case_3CAV1HV_{}'.format(current_time) + '/DATA_REAL.txt')
            save_varible(self.DATA_CAL, './Case_3CAV1HV/'+'Case_3CAV1HV_{}'.format(current_time) + '/DATA_CAL.txt')
            save_varible(self.DATA_THETA, './Case_3CAV1HV/'+'Case_3CAV1HV_{}'.format(current_time) + '/DATA_THETA.txt')
            save_varible(self.TIME_RECORD, './Case_3CAV1HV/'+'Case_3CAV1HV_{}'.format(current_time) + '/TIME_RECORD.txt')
            await asyncio.sleep(0.5)

    async def task2(self):
        print('执行任务2——AVCONTROL')
        while True:
            # print('self.RESULT', self.RESULT)
            # result = await result_queue.get()
            if self.RESULT is None:
                pass
            else:
                self.av_control(self.RESULT)
                
            rclpy.spin_once(self)
            await asyncio.sleep(0.01)

    async def run(self):
        while rclpy.ok():
            # result_queue = asyncio.Queue()
            # task1_coroutine = asyncio.create_task(self.task1())
            # task2_coroutine = asyncio.create_task(self.task2())
            # await task1_coroutine
            # await task2_coroutine


            await asyncio.gather(self.task1(), self.task2())
        
    # def run(self):
        
    #     while rclpy.ok():
    #         self.DATA_REAL = load_variable('DATA_REAL.txt')
    #         self.DATA_CAL = load_variable('DATA_CAL.txt')
    #         # print(self.DATA_REAL)
    #         time_1 = time.time()
    #         result = self.compute()
    #         time_2 = time.time()
    #         print('整体计算COMPUTE模块耗时{}'.format(time_2-time_1))
    #         if result == None:
    #             if self.RESULT == None:
    #                 pass
    #             else:
    #                 self.av_control(self.RESULT)
    #         else:
    #             self.av_control(result)
    #         rclpy.spin_once(self)
    #         # print(self.DATA_REAL)
    #         save_varible(self.DATA_REAL,'DATA_REAL.txt')
    #         save_varible(self.DATA_CAL,'DATA_CAL.txt')


def main(args=None):
    rclpy.init(args=args)
    node = sub_pub()
    asyncio.run(node.run())
    node.destroy_node()
    rclpy.shutdown()



if __name__ == '__main__':
    main()
