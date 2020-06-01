import numpy as np

class World:
    def __init__(self):

        # 地图参数
        self.map_width = 10000  # 单位m，若单位为unit个数，则需要改变数值
        self.map_length = 10000
        self.map_height = 500  # 可以暂时不考虑，只考虑二维
        self.num_Users = 8  # 用户个数
        self.num_UAVs = 2   # 无人机个数
        self.num_GBS = 1    # 地面基站个数
        self.pos_Users = [] # 用户位置坐标，shape:num_Users*3
        self.pos_UAVs = []  # 无人机位置坐标，shape：num_UAVs* 3
        self.pos_GBS = []   # 地面基站位置坐标，shape：num_GBS*3
        self.vel_Users = None    # 用户移动速度
        self.vel_UAV = None      # 无人机移动速度
        self.max_vel_UAV = 30     # 无人机最大速度
        self.max_vel_User = 2  # 用户最大速度
        self.angle_Users = None  # 用户移动方向
        self.angle_UAVs = None   # 无人机移动方向
        self.pow_UAVs = None    # 无人机发射功率
        self.pow_GBS = None   # 地面基站发射功率
        self.max_pow = 1000 # 最大发射功率 mW
        self.min_pow = 0  # 最小发射功率 mW
        self.max_energy_consume = 60000  # 最大能耗
        self.sim_period = 600
        self.dt = 0.1   # 时间步间隔
        self.gain_UAV = 40  # 单位db 无人机链路单位距离下的信道增益
        self.gain_GBS = 60  # 单位db 地面基站链路单位距离下的信道增益
        self.los_exp_UAV = 2  # 无人机链路衰减指数
        self.los_exp_GBS = 3  # 地面基站链路衰减指数
        self.bandwidth = None  # 带宽
        self.max_band = 10e6   # 最大可用带宽
        self.noise_pdf = -167  # 噪声功率谱密度 dBm

        self.dim_vel = 1  # 速度动作维度，一维
        self.dim_ang = 1   # 方向动作维度，一维
        self.dim_alloc = self.num_Users   # 调度动作维度




