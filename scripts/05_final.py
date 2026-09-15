"""
工业级 10 维物理动力学与接触隐藏参数 Sim-to-Sim 标定系统 (互相关锁定+交替优化版)
=====================================================================
- 基于原始代码，仅增加：
  1. 互相关法估计系统时延（sys_delay），避免优化器搜索
  2. 两轮交替优化（Refine B, Refine C），消除参数代偿
- 保持 Stage B 和 Stage C 原仿真方法不变
"""

import time
import numpy as np
import pybullet as p
import pybullet_data
import optuna
import matplotlib.pyplot as plt
from collections import deque

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =====================================================================
# 1. 参数空间映射与对数尺度归一化器
# =====================================================================
class ParamNormalizer:
    def __init__(self, bounds):
        self.bounds = bounds
        self.log_params = {'k_n', 'c_n'}
        self.param_names = list(bounds.keys())

    def norm_to_phys(self, name, val_norm):
        low, high = self.bounds[name]
        val_norm = np.clip(val_norm, 0.0, 1.0)
        if name in self.log_params:
            log_low, log_high = np.log(low), np.log(high)
            return float(np.exp(log_low + val_norm * (log_high - log_low)))
        else:
            return float(low + val_norm * (high - low))

    def phys_to_norm(self, name, val_phys):
        low, high = self.bounds[name]
        if name in self.log_params:
            log_low, log_high = np.log(low), np.log(high)
            val_phys = np.clip(val_phys, low, high)
            return float((np.log(val_phys) - log_low) / (log_high - log_low))
        else:
            val_phys = np.clip(val_phys, low, high)
            return float((val_phys - low) / (high - low))

    def transform_vector(self, x_norm):
        phys_dict = {}
        for idx, key in enumerate(self.param_names):
            phys_dict[key] = self.norm_to_phys(key, x_norm[idx])
        return phys_dict


# =====================================================================
# 2. 物理仿真器
# =====================================================================
class RobotSimulator:
    def __init__(self, gui=False):
        self.gui = gui #GUI 模式会弹出一个 3D 可视化窗口（方便调试），DIRECT 模式是后台无头渲染
        self.physics_client = None
        self.robot_id = None
        self.cube_id = None
        self.plane_id = None
        self.EE_INDEX = 11  # EE_INDEX 是你代码中定义的末端执行器（End-Effector）的关节索引11 对应的是 panda_grasptip（夹爪指尖/末端工具坐标系）
        self.FINGER_L, self.FINGER_R = 9, 10
        self.ARM_JOINTS = list(range(7))
        self.use_external_torque_com = False
        self.action_buffer = deque()

    def _extract_force(self, val):
        """从接触点数据中提取标量力值，若为向量则取模"""
        if isinstance(val, (tuple, list)):
            return float(np.linalg.norm(val))
        return float(val)

    def _get_static_wrist_torque(self, duration=0.3):
        """在当前位置静止采集腕部关节（5,6）平均力矩，用于质量标定基准"""
        steps = int(duration * 240)
        torques = []
        for _ in range(steps):
            p.stepSimulation()
            states = p.getJointStates(self.robot_id, [5, 6])
            torques.append([states[0][3], states[1][3]])  # 应用力矩
        return np.mean(torques, axis=0)  # 返回 [tau5_mean, tau6_mean]

    # 🔧 新增：互相关估计系统延迟（单位：帧）
    def estimate_sys_delay(self, params, duration=0.5, joint_idx=4):
        """
        使用阶跃响应估计系统延迟（帧数）
        """
        self.set_params(params)

        dt = 1.0 / 240.0
        steps = int(duration * 240)

        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        base_j = list(p.calculateInverseKinematics(self.robot_id, self.EE_INDEX, [0.5, 0.0, 0.3], target_orn, currentPositions=[0.0] * 9, maxNumIterations=100))
        #我要让机械臂的指尖（EE_INDEX=11）移动到空间坐标 [0.5, 0.0, 0.3] 米处，并且末端要朝下（旋转矩阵为 [π,0,0]）
        # 立方体放远并底面贴地：避免穿透地面产生 k_n 巨力弹飞（曾污染延迟估计）
        self.reset_to_state(cube_pos=[1.5, 0.0, 0.025], arm_j=base_j, finger_pos=0.04)

        # 初始化缓冲，保持静止，清空动作缓存并填充基线指令（防止缓存空导致首次执行异常）
        self.action_buffer.clear()
        for _ in range(self.delay_steps):
            self.action_buffer.append((base_j, 0.04, 5.0))

        # 阶跃命令：前 30 帧保持基线，然后突然改变关节 4 目标
        step_trigger = 30
        step_target = base_j[joint_idx] + 0.2  # 阶跃幅度：关节 4 上施加一个固定阶跃 0.2 rad

        cmd_traj = []
        vel_traj = []

        for i in range(steps):
            curr_cmd = list(base_j)
            if i >= step_trigger:
                curr_cmd[joint_idx] = step_target
            else:
                curr_cmd[joint_idx] = base_j[joint_idx]

            self._apply_action(curr_cmd, finger_pos=0.04, finger_force=5.0, arm_force=120.0)
            p.stepSimulation()

            cmd_traj.append(curr_cmd[joint_idx]) #记录指令曲线
            state = p.getJointState(self.robot_id, joint_idx)
            vel_traj.append(state[1])#因为位置信号在阶跃初期变化非常缓慢（微分滞后），而速度信号对阶跃指令的响应是瞬间跳变的。

        # 找到速度开始偏离基线的时刻（响应开始）
        baseline_vel = np.mean(vel_traj[:step_trigger])
        vel_arr = np.array(vel_traj)
        threshold = baseline_vel + 0.05 * np.max(np.abs(vel_arr))  # 阈值设为峰值5%
        # 从阶跃时刻往后搜索，找到第一组三个都超过阈值的点（防噪音）
        consecutive_count = 0
        response_start = None
        for i in range(step_trigger, steps):
            if abs(vel_arr[i] - baseline_vel) > threshold:
                consecutive_count += 1
                if consecutive_count >= 3:
                    response_start = i - 2  # 取连续第一次超过的时刻
                    break
            else:
                consecutive_count = 0
        if response_start is None:
            response_start = steps - 1

        delay_frames = max(0, response_start - step_trigger)#防止负数
        return delay_frames

    def connect(self):
        if self.physics_client is not None:
            return   #如果已经连上了，就不再重复连接
        self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(
            numSolverIterations=150,
            numSubSteps=4,
            enableConeFriction=1,
            contactBreakingThreshold=0.001
        )#之前调好的参数设置
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0.0, 0.2])

        for i in range(p.getNumJoints(self.robot_id)):
            p.enableJointForceTorqueSensor(self.robot_id, i, enableSensor=1)

    def disconnect(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

    def reset_to_state(self, cube_pos=(0.5, 0.0, 0.2), arm_j=None, finger_pos=0.04):
        p.resetBasePositionAndOrientation(self.cube_id, cube_pos, (0,0,0,1))
        p.resetBaseVelocity(self.cube_id, [0,0,0], [0,0,0])
        if arm_j is None:
            target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
            arm_j = p.calculateInverseKinematics(self.robot_id, self.EE_INDEX, cube_pos, target_orn, currentPositions=[0.0] * 9, maxNumIterations=100)
        for idx, j_idx in enumerate(self.ARM_JOINTS):
            p.resetJointState(self.robot_id, j_idx, arm_j[idx], targetVelocity=0.0)
        p.resetJointState(self.robot_id, self.FINGER_L, finger_pos, targetVelocity=0.0)
        p.resetJointState(self.robot_id, self.FINGER_R, finger_pos, targetVelocity=0.0)

    def set_params(self, params):
        p.changeDynamics(
            self.cube_id, -1,
            mass=params['mass'],
            lateralFriction=params['mu_lat'],
            spinningFriction=params['mu_spin'],
            contactStiffness=params['k_n'],
            contactDamping=params['c_n']
        )
        p.changeDynamics(self.robot_id, self.FINGER_L, lateralFriction=params['mu_lat'], spinningFriction=params['mu_spin'])
        p.changeDynamics(self.robot_id, self.FINGER_R, lateralFriction=params['mu_lat'], spinningFriction=params['mu_spin'])
        for j in self.ARM_JOINTS:
            p.changeDynamics(self.robot_id, j, jointDamping=params['joint_damp'])

        try:
            p.changeDynamics(
                self.cube_id, -1,
                localInertialFramePosition=[params['com_dx'], params['com_dy'], params['com_dz']],
                localInertialFrameOrientation=[0,0,0,1]
            )
            self.use_external_torque_com = False
        except TypeError:
            self.use_external_torque_com = True

        self.delay_steps = max(1, int(round(params['sys_delay'] * 240.0)))
        self.action_buffer = deque(maxlen=self.delay_steps)

    def _apply_action(self, target_j, finger_pos, finger_force, arm_force=150.0):
        self.action_buffer.append((target_j, finger_pos, finger_force))
        exec_j, exec_finger_pos, exec_finger_force = self.action_buffer[0]

        for idx, j_idx in enumerate(self.ARM_JOINTS):
            # VELOCITY_CONTROL：POSITION_CONTROL 对这个 Panda 臂（roll=π 姿态）会下垂/振荡抓不住，
            # 改用速度指令 + 强刹车力，让臂能稳定持姿并跟踪旋转
            current = p.getJointState(self.robot_id, j_idx)[0]
            target_vel = (exec_j[idx] - current) * 15.0
            p.setJointMotorControl2(
                self.robot_id, j_idx, p.VELOCITY_CONTROL,
                targetVelocity=target_vel,
                force=5000.0
            )
        p.setJointMotorControl2(
            self.robot_id, self.FINGER_L, p.POSITION_CONTROL,
            targetPosition=exec_finger_pos, force=exec_finger_force
        )
        p.setJointMotorControl2(
            self.robot_id, self.FINGER_R, p.POSITION_CONTROL,
            targetPosition=exec_finger_pos, force=exec_finger_force
        )

    def _settle_grasp(self, hold_pos=(0.5, 0.0, 0.2), finger_target=0.015, finger_force=12.0, steps=60):
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        init_j = list(p.calculateInverseKinematics(self.robot_id, self.EE_INDEX, hold_pos, target_orn, currentPositions=[0.0] * 9, maxNumIterations=100))
        self.reset_to_state(cube_pos=hold_pos, arm_j=init_j, finger_pos=0.025)
        self.action_buffer.clear()
        for _ in range(self.delay_steps):
            self.action_buffer.append((init_j, finger_target, finger_force))
        for _ in range(steps):
            # 抓取建立阶段：固定机械臂位姿并托住立方体，仅让夹爪闭合（避免机械臂下垂 & 立方体自由落体）
            for idx, j_idx in enumerate(self.ARM_JOINTS):
                p.resetJointState(self.robot_id, j_idx, init_j[idx], targetVelocity=0.0)
            p.resetBaseVelocity(self.cube_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            self._apply_action(init_j, finger_pos=finger_target, finger_force=finger_force)
            p.stepSimulation()

    def _measure_static_grasp_torque(self, params, duration=0.15):
        """抓取立方体后静止采集腕关节(5,6)平均力矩，用于质量标定（只依赖 mass，与 k_n/c_n 解耦）"""
        self.set_params(params)
        self._settle_grasp(hold_pos=(0.5, 0.0, 0.2), finger_target=0.018, finger_force=15.0, steps=50)
        steps = int(duration * 240)
        torques = []
        for _ in range(steps):
            for idx, j_idx in enumerate(self.ARM_JOINTS):
                p.setJointMotorControl2(self.robot_id, j_idx, p.VELOCITY_CONTROL, targetVelocity=0.0, force=5000.0)
            p.setJointMotorControl2(self.robot_id, self.FINGER_L, p.POSITION_CONTROL, targetPosition=0.018, force=15.0)
            p.setJointMotorControl2(self.robot_id, self.FINGER_R, p.POSITION_CONTROL, targetPosition=0.018, force=15.0)
            p.stepSimulation()
            states = p.getJointStates(self.robot_id, [5, 6])
            torques.append([states[0][3], states[1][3]])
        return np.mean(torques, axis=0)

    # -----------------------------------------------------------------
    # 场景 A：空载多频扫频与阶跃 (保持原样)
    # -----------------------------------------------------------------
    def simulate_stage_A(self, params, duration=2.5):
        self.set_params(params)
        dt = 1.0 / 240.0
        steps = int(duration * 240)
        times = np.linspace(0, duration, steps)

        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        base_j = list(p.calculateInverseKinematics(self.robot_id, self.EE_INDEX, [0.5, 0.0, 0.3], target_orn, currentPositions=[0.0] * 9, maxNumIterations=100))
        # 立方体放远并底面贴地：避免穿透地面产生 k_n 巨力弹飞撞臂（曾污染 Stage A 力矩信号）
        self.reset_to_state(cube_pos=[1.5, 0.0, 0.025], arm_j=base_j, finger_pos=0.04)
        self.action_buffer.clear()
        for _ in range(self.delay_steps):
            self.action_buffer.append((base_j, 0.04, 5.0))

        freq = np.linspace(0.5, 3.5, steps)
        phase = 2 * np.pi * np.cumsum(freq) * dt
        cmd_j4 = base_j[4] + 0.35 * np.sin(phase)
        cmd_j3 = base_j[3] + 0.20 * np.cos(phase * 0.7)

        joint_pos_list, joint_vel_list, joint_torque_list = [], [], []

        for i in range(steps):
            curr_cmd = list(base_j)
            curr_cmd[4] = cmd_j4[i]
            curr_cmd[3] = cmd_j3[i]
            self._apply_action(curr_cmd, finger_pos=0.04, finger_force=5.0, arm_force=120.0)
            p.stepSimulation()
            states = p.getJointStates(self.robot_id, self.ARM_JOINTS)
            joint_pos_list.append([s[0] for s in states])
            joint_vel_list.append([s[1] for s in states])
            joint_torque_list.append([s[2][3:6] for s in states])

        step_cmd = list(base_j)
        step_cmd[4] = base_j[4]
        step_cmd[3] = base_j[3]
        for _ in range(80):
            self._apply_action(step_cmd, finger_pos=0.04, finger_force=5.0, arm_force=120.0)
            p.stepSimulation()
            states = p.getJointStates(self.robot_id, self.ARM_JOINTS)
            joint_pos_list.append([s[0] for s in states])
            joint_vel_list.append([s[1] for s in states])
            joint_torque_list.append([s[2][3:6] for s in states])

        return np.array(joint_pos_list), np.array(joint_vel_list), np.array(joint_torque_list)

    # -----------------------------------------------------------------
    # 场景 B：落块冲击实验（自由落体撞击平面，动态编码 k_n/c_n）
    # -----------------------------------------------------------------
    def simulate_stage_B(self, params, duration=1.0):
        self.set_params(params)
        steps = int(duration * 240)

        # 夹爪回到默认位姿，远离立方体下落路径；立方体从 0.3m 自由落体撞击平面
        base_j = [0.0] * 7
        self.reset_to_state(cube_pos=(0.5, 0.0, 0.3), arm_j=base_j, finger_pos=0.04)
        self.action_buffer.clear()
        for _ in range(self.delay_steps):
            self.action_buffer.append((base_j, 0.04, 5.0))

        cube_z_list, normal_force_list, wrist_torque_list = [], [], []

        for i in range(steps):
            self._apply_action(base_j, finger_pos=0.04, finger_force=5.0, arm_force=120.0)
            p.stepSimulation()

            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            cube_z_list.append(cube_pos[2])

            contacts = p.getContactPoints(self.cube_id, self.plane_id)
            if contacts:
                nf = max([self._extract_force(c[9]) for c in contacts])
                normal_force_list.append(float(nf))
            else:
                normal_force_list.append(0.0)

            wrist_states = p.getJointStates(self.robot_id, [5, 6])
            wrist_torque_list.append([wrist_states[0][3], wrist_states[1][3]])

        return np.array(cube_z_list), np.array(normal_force_list), np.array(wrist_torque_list)

    # -----------------------------------------------------------------
    # 场景 C：复合多轴旋转与微滑移 (保持原样，不做任何修改)
    # -----------------------------------------------------------------
    def simulate_stage_C(self, params, duration=3.0):
        self.set_params(params)
        steps = int(duration * 240)
        times = np.linspace(0, duration, steps)

        self._settle_grasp(hold_pos=(0.5, 0.0, 0.2), finger_target=0.020, finger_force=3.5, steps=50)

        roll = np.pi + 0.6 * np.sin(2 * np.pi * 0.6 * times)
        pitch = 0.4 * np.sin(2 * np.pi * 0.9 * times)
        yaw = 0.3 * np.sin(2 * np.pi * 1.5 * times)

        cube_pos_list, cube_orn_list, contact_force_list = [], [], []

        for i in range(steps):
            orn = p.getQuaternionFromEuler([roll[i], pitch[i], yaw[i]])
            cmd_j = p.calculateInverseKinematics(self.robot_id, self.EE_INDEX, [0.5, 0.0, 0.20], orn, maxNumIterations=100)

            self._apply_action(cmd_j, finger_pos=0.020, finger_force=3.5, arm_force=150.0)

            if self.use_external_torque_com:
                pos, orn_cube = p.getBasePositionAndOrientation(self.cube_id)
                rot_mat = np.array(p.getMatrixFromQuaternion(orn_cube)).reshape(3, 3)
                local_com = np.array([params['com_dx'], params['com_dy'], params['com_dz']])
                world_com_offset = rot_mat.dot(local_com)
                gravity_force = np.array([0, 0, -9.81 * params['mass']])
                world_torque = np.cross(world_com_offset, gravity_force)
                p.applyExternalTorque(self.cube_id, -1, world_torque, p.WORLD_FRAME)

            p.stepSimulation()

            pos, orn_cube = p.getBasePositionAndOrientation(self.cube_id)
            cube_pos_list.append([float(pos[0]), float(pos[1]), float(pos[2])])
            cube_orn_list.append(list(map(float, p.getEulerFromQuaternion(orn_cube))))

            contacts = p.getContactPoints(self.robot_id, self.cube_id)
            if contacts:
                nf = max([self._extract_force(c[9]) for c in contacts])
                lat1 = max([self._extract_force(c[10]) for c in contacts])
                lat2 = max([self._extract_force(c[12]) for c in contacts])  # 索引修正
                contact_force_list.append([nf, lat1, lat2])
            else:
                contact_force_list.append([0.0, 0.0, 0.0])

        return np.array(cube_pos_list), np.array(cube_orn_list), np.array(contact_force_list)

    # -----------------------------------------------------------------
    # 场景 S：平面扭转衰减实验（绕接触法线旋转，动态编码 mu_spin）
    # -----------------------------------------------------------------
    def simulate_stage_S(self, params, duration=0.2):
        """立方体平放平面，绕 z 轴(接触法线)给初始角速度，记录角速度衰减曲线。
        衰减率 ∝ spinningFriction，是干净的 mu_spin 信号（与夹持/质心解耦）。"""
        self.set_params(params)
        steps = int(duration * 240)

        base_j = [0.0] * 7  # 手臂默认位姿，远离立方体下落/旋转路径
        self.reset_to_state(cube_pos=(0.5, 0.0, 0.025), arm_j=base_j, finger_pos=0.04)
        self.action_buffer.clear()
        for _ in range(self.delay_steps):
            self.action_buffer.append((base_j, 0.04, 5.0))

        # settle：让立方体稳定接触平面
        for _ in range(60):
            self._apply_action(base_j, finger_pos=0.04, finger_force=5.0)
            p.stepSimulation()

        # 给绕 z 轴初始角速度，随后测量其衰减
        p.resetBaseVelocity(self.cube_id, [0, 0, 0], [0, 0, 10.0])

        omega_z_list = []
        for _ in range(steps):
            self._apply_action(base_j, finger_pos=0.04, finger_force=5.0)
            p.stepSimulation()
            _lin, ang = p.getBaseVelocity(self.cube_id)
            omega_z_list.append(ang[2])

        return np.array(omega_z_list)


# =====================================================================
# 3. 损失函数
# =====================================================================
def moving_average(signal, window=5):
    """沿时间轴(axis=0)做滑动平均滤波，抑制接触力高频震荡"""
    signal = np.asarray(signal, dtype=float)
    kernel = np.ones(window) / window
    if signal.ndim == 1:
        return np.convolve(signal, kernel, mode='same')
    if signal.ndim == 2:
        out = np.empty_like(signal)
        for col in range(signal.shape[1]):
            out[:, col] = np.convolve(signal[:, col], kernel, mode='same')
        return out
    return signal

def compute_loss_stage_A(sim_data, real_data, scales):
    sim_q, sim_v, sim_t = sim_data
    real_q, real_v, real_t = real_data
    assert sim_q.shape == real_q.shape, "Stage A 位置维度不一致"
    assert sim_v.shape == real_v.shape, "Stage A 速度维度不一致"
    assert sim_t.shape == real_t.shape, "Stage A 力矩维度不一致"
    loss_q = np.mean((sim_q - real_q)**2) / scales['q']
    loss_v = np.mean((sim_v - real_v)**2) / scales['v']
    loss_t = np.mean((sim_t - real_t)**2) / scales['t']
    return float(loss_q + loss_v + loss_t)

def compute_loss_stage_B(sim_data, real_data, scales):
    sim_z, sim_nf, sim_tq = sim_data
    real_z, real_nf, real_tq = real_data
    assert sim_z.shape == real_z.shape, "Stage B 高度维度不一致"
    assert sim_nf.shape == real_nf.shape, "Stage B 接触力维度不一致"
    assert sim_tq.shape == real_tq.shape, "Stage B 腕力矩维度不一致"
    # 接触力信号先做滑动平均滤波，抑制高频震荡
    sim_nf = moving_average(sim_nf, window=5)
    real_nf = moving_average(real_nf, window=5)
    loss_z = np.mean((sim_z - real_z)**2) / scales['z']
    loss_nf = np.mean((sim_nf - real_nf)**2) / scales['nf']
    loss_tq = np.mean((sim_tq - real_tq)**2) / scales['tq']
    return float(loss_z + loss_nf + loss_tq)

def compute_loss_stage_C(sim_data, real_data, scales, com_xyz=(0.0, 0.0, 0.0)):
    sim_pos, sim_orn, sim_cf = sim_data
    real_pos, real_orn, real_cf = real_data
    assert sim_pos.shape == real_pos.shape, "Stage C 位置维度不一致"
    assert sim_orn.shape == real_orn.shape, "Stage C 姿态维度不一致"
    assert sim_cf.shape == real_cf.shape, "Stage C 接触力维度不一致"
    # 接触力信号先做滑动平均滤波，抑制高频震荡
    sim_cf = moving_average(sim_cf, window=5)
    real_cf = moving_average(real_cf, window=5)
    loss_pos = np.mean((sim_pos - real_pos)**2) / scales['pos']
    loss_orn = np.mean((sim_orn - real_orn)**2) / scales['orn']
    loss_cf = np.mean((sim_cf - real_cf)**2) / scales['cf']
    # 几何正则化：防止质心漂到搜索边界（com 米制转 mm 才让 0.001 生效）
    reg = 0.001 * ((com_xyz[0] * 1000)**2 + (com_xyz[1] * 1000)**2 + (com_xyz[2] * 1000)**2)
    return float(loss_pos + loss_orn + loss_cf + reg)


def compute_loss_stage_S(sim_omega, real_omega, scale):
    assert sim_omega.shape == real_omega.shape, "Stage S 角速度维度不一致"
    return float(np.mean((sim_omega - real_omega) ** 2) / scale)


# =====================================================================
# 4. 主程序（新增互相关锁定与两轮交替优化）
# =====================================================================
def main():
    np.random.seed(42)
    print("=" * 70)
    print("🔧 [工业级物理参数辨识] 10 维全参数标定 (互相关+交替优化)")
    print("=" * 70)

    TRUE_PARAMS = {
        "mass": 0.15,
        "mu_lat": 0.80,
        "mu_spin": 0.05,
        "k_n": 5000.0,
        "c_n": 50.0,
        "com_dx": 0.005,
        "com_dy": -0.005,
        "com_dz": 0.002,
        "joint_damp": 1.5,
        "sys_delay": 0.03
    }

    PARAM_BOUNDS = {
        "mass": (0.08, 0.25),
        "mu_lat": (0.40, 1.10),
        "mu_spin": (0.025, 0.1),
        "k_n": (2000.0, 10000.0),
        "c_n": (20.0, 120.0),
        "com_dx": (-0.008, 0.008),
        "com_dy": (-0.008, 0.008),
        "com_dz": (-0.008, 0.008),
        "joint_damp": (0.2, 3.5),
        "sys_delay": (0.0, 0.08)
    }

    NOMINAL = {
        "mass": 0.10, "mu_lat": 0.40, "mu_spin": 0.02,
        "k_n": 2000.0, "c_n": 20.0,
        "com_dx": 0.0, "com_dy": 0.0, "com_dz": 0.0,
        "joint_damp": 0.5, "sys_delay": 0.01
    }

    normalizer = ParamNormalizer(PARAM_BOUNDS)

    print("\n📡 [Step 0] 互相关估计系统延迟...")
    sim = RobotSimulator(gui=False)
    sim.connect()

    try:
        # 🔧 互相关估计 sys_delay（使用真实参数）
        delay_frames_est = sim.estimate_sys_delay(TRUE_PARAMS)
        est_sys_delay = delay_frames_est / 240.0
        TRUE_PARAMS['sys_delay'] = est_sys_delay  # 更新真值，使后续真实数据与估计一致
        NOMINAL['sys_delay'] = est_sys_delay      # 固定到名义值
        # 估计前
        print(f"  真实延迟: {TRUE_PARAMS['sys_delay']*240:.1f} 帧 = {TRUE_PARAMS['sys_delay']:.4f} s")
        # 估计后
        print(f"  估计延迟: {delay_frames_est} 帧 = {est_sys_delay:.4f} s")

        print("\n📡 [Step 1a] 生成 Stage A 真实轨迹...")
        real_A = sim.simulate_stage_A(TRUE_PARAMS)

        # 添加传感器噪声（噪声已降低，避免淹没微弱物理特征）
        real_A_noisy = (
            real_A[0] + np.random.normal(0, 1e-5, real_A[0].shape),
            real_A[1] + np.random.normal(0, 0.005, real_A[1].shape),
            real_A[2] + np.random.normal(0, 0.02, real_A[2].shape)
        )
        scales_A = {
            'q': np.var(real_A_noisy[0]) + 1e-5,
            'v': np.var(real_A_noisy[1]) + 1e-5,
            't': np.var(real_A_noisy[2]) + 1e-5
        }

        loss_history = {}

        def make_sampler(n_startup=15):
            # Optuna 4.9.0：sigma0 / restart_strategy 已弃用，步长由归一化参数空间缩放控制
            return optuna.samplers.CmaEsSampler(seed=42, n_startup_trials=n_startup)

        # -------------------------------------------------------------
        # Stage A：只优化 joint_damp (sys_delay 已固定)
        # -------------------------------------------------------------
        print("\n📘 [Stage A] 辨识 joint_damp (sys_delay 已锁定)...")
        def objective_A(trial):
            p_dict = NOMINAL.copy()
            p_dict['joint_damp'] = normalizer.norm_to_phys('joint_damp', trial.suggest_float('joint_damp', 0, 1))
            sim_A = sim.simulate_stage_A(p_dict)
            return compute_loss_stage_A(sim_A, real_A_noisy, scales_A)

        study_A = optuna.create_study(sampler=make_sampler(15), direction="minimize")
        study_A.optimize(objective_A, n_trials=60, show_progress_bar=False)
        loss_history['Stage A'] = study_A.trials_dataframe()['value'].tolist()
        NOMINAL['joint_damp'] = normalizer.norm_to_phys('joint_damp', study_A.best_params['joint_damp'])
        print(f"  --> 阶段 A 完成: joint_damp={NOMINAL['joint_damp']:.4f}")

        # -------------------------------------------------------------
        # Step 1b：生成 Stage B/C 真实数据（放在 Stage A 之后，避免落块实验污染 joint_damp 辨识）
        # -------------------------------------------------------------
        print("\n📡 [Step 1b] 生成 Stage B/C/S 真实轨迹...")
        real_B = sim.simulate_stage_B(TRUE_PARAMS)
        real_C = sim.simulate_stage_C(TRUE_PARAMS)
        real_S = sim.simulate_stage_S(TRUE_PARAMS)
        clean_data = {'A': real_A, 'B': real_B, 'C': real_C, 'S': real_S}
        real_B_noisy = (
            real_B[0] + np.random.normal(0, 1e-5, real_B[0].shape),
            real_B[1] + np.random.normal(0, 0.01, real_B[1].shape),
            real_B[2] + np.random.normal(0, 0.01, real_B[2].shape)
        )
        real_C_noisy = (
            real_C[0] + np.random.normal(0, 1e-5, real_C[0].shape),
            real_C[1] + np.random.normal(0, 1e-5, real_C[1].shape),
            real_C[2] + np.random.normal(0, 0.01, real_C[2].shape)
        )
        scales_B = {
            'z': np.var(real_B_noisy[0]) + 1e-5,
            'nf': np.var(real_B_noisy[1]) + 1e-5,
            'tq': np.var(real_B_noisy[2]) + 1e-5
        }
        scales_C = {
            'pos': np.var(real_C_noisy[0]) + 1e-5,
            'orn': np.var(real_C_noisy[1]) + 1e-5,
            'cf': np.var(real_C_noisy[2]) + 1e-5
        }
        real_S_noisy = real_S + np.random.normal(0, 0.01, real_S.shape)
        scales_S = {'omega': np.var(real_S_noisy) + 1e-5}

        # -------------------------------------------------------------
        # Stage M：用静止腕力矩单独标定 mass（与 k_n/c_n 解耦）
        # -------------------------------------------------------------
        print("\n📘 [Stage M] 用静止腕力矩标定 mass...")
        real_static_tau5 = sim._measure_static_grasp_torque(TRUE_PARAMS)[0]
        def objective_M(trial):
            p_dict = NOMINAL.copy()
            p_dict['mass'] = normalizer.norm_to_phys('mass', trial.suggest_float('mass', 0, 1))
            tau5 = sim._measure_static_grasp_torque(p_dict)[0]
            return float((tau5 - real_static_tau5) ** 2)

        study_M = optuna.create_study(sampler=make_sampler(15), direction="minimize")
        study_M.optimize(objective_M, n_trials=40, show_progress_bar=False)
        loss_history['Stage M'] = study_M.trials_dataframe()['value'].tolist()
        NOMINAL['mass'] = normalizer.norm_to_phys('mass', study_M.best_params['mass'])
        print(f"  --> 阶段 M 完成: mass={NOMINAL['mass']:.4f}")

        # -------------------------------------------------------------
        # Stage B：辨识 mass, k_n, c_n (原方法)
        # -------------------------------------------------------------
        print("\n📘 [Stage B] 辨识接触参数: k_n, c_n (mass 已固定)...")
        def objective_B(trial):
            p_dict = NOMINAL.copy()
            p_dict['k_n'] = normalizer.norm_to_phys('k_n', trial.suggest_float('k_n', 0, 1))
            p_dict['c_n'] = normalizer.norm_to_phys('c_n', trial.suggest_float('c_n', 0, 1))
            sim_B = sim.simulate_stage_B(p_dict)
            return compute_loss_stage_B(sim_B, real_B_noisy, scales_B)

        study_B = optuna.create_study(sampler=make_sampler(20), direction="minimize")
        study_B.optimize(objective_B, n_trials=150, show_progress_bar=False)
        loss_history['Stage B'] = study_B.trials_dataframe()['value'].tolist()
        NOMINAL['k_n'] = normalizer.norm_to_phys('k_n', study_B.best_params['k_n'])
        NOMINAL['c_n'] = normalizer.norm_to_phys('c_n', study_B.best_params['c_n'])
        print(f"  --> 阶段 B 完成: k_n={NOMINAL['k_n']:.1f}, c_n={NOMINAL['c_n']:.2f}")

        # -------------------------------------------------------------
        # Stage S：平面扭转衰减单独标定 mu_spin（与夹持摩擦/质心解耦）
        # -------------------------------------------------------------
        print("\n📘 [Stage S] 用平面扭转衰减标定 mu_spin...")
        def objective_S(trial):
            p_dict = NOMINAL.copy()
            p_dict['mu_spin'] = normalizer.norm_to_phys('mu_spin', trial.suggest_float('mu_spin', 0, 1))
            sim_S = sim.simulate_stage_S(p_dict)
            return compute_loss_stage_S(sim_S, real_S_noisy, scales_S['omega'])

        study_S = optuna.create_study(sampler=make_sampler(15), direction="minimize")
        study_S.optimize(objective_S, n_trials=80, show_progress_bar=False)
        loss_history['Stage S'] = study_S.trials_dataframe()['value'].tolist()
        NOMINAL['mu_spin'] = normalizer.norm_to_phys('mu_spin', study_S.best_params['mu_spin'])
        print(f"  --> 阶段 S 完成: mu_spin={NOMINAL['mu_spin']:.4f}")

        # -------------------------------------------------------------
        # Stage C：辨识摩擦与质心偏移 (mu_spin 已由 Stage S 标定)
        # -------------------------------------------------------------
        print("\n📘 [Stage C] 辨识表面摩擦与 3D 质心偏移 (mu_lat, com_xyz)...")
        def objective_C(trial):
            p_dict = NOMINAL.copy()
            for key in ['mu_lat', 'com_dx', 'com_dy', 'com_dz']:
                p_dict[key] = normalizer.norm_to_phys(key, trial.suggest_float(key, 0, 1))
            sim_C = sim.simulate_stage_C(p_dict)
            com_xyz = (p_dict['com_dx'], p_dict['com_dy'], p_dict['com_dz'])
            return compute_loss_stage_C(sim_C, real_C_noisy, scales_C, com_xyz=com_xyz)

        study_C = optuna.create_study(sampler=make_sampler(30), direction="minimize")
        study_C.optimize(objective_C, n_trials=300, show_progress_bar=False)
        loss_history['Stage C'] = study_C.trials_dataframe()['value'].tolist()
        for key in ['mu_lat', 'com_dx', 'com_dy', 'com_dz']:
            NOMINAL[key] = normalizer.norm_to_phys(key, study_C.best_params[key])
        print(f"  --> 阶段 C 完成: mu_lat={NOMINAL['mu_lat']:.4f}, com_dz={NOMINAL['com_dz']:.4f}")

        # -------------------------------------------------------------
        # 🔧 第二轮精修：Refine B (固定摩擦与质心)
        # -------------------------------------------------------------
        print("\n🔁 [Refine B] 精修 k_n, c_n (mass 固定)...")
        def objective_B_refine(trial):
            p_dict = NOMINAL.copy()
            p_dict['k_n'] = normalizer.norm_to_phys('k_n', trial.suggest_float('k_n', 0, 1))
            p_dict['c_n'] = normalizer.norm_to_phys('c_n', trial.suggest_float('c_n', 0, 1))
            sim_B = sim.simulate_stage_B(p_dict)
            return compute_loss_stage_B(sim_B, real_B_noisy, scales_B)

        study_B_ref = optuna.create_study(sampler=make_sampler(20), direction="minimize")
        study_B_ref.optimize(objective_B_refine, n_trials=150, show_progress_bar=False)
        loss_history['Refine B'] = study_B_ref.trials_dataframe()['value'].tolist()
        NOMINAL['k_n'] = normalizer.norm_to_phys('k_n', study_B_ref.best_params['k_n'])
        NOMINAL['c_n'] = normalizer.norm_to_phys('c_n', study_B_ref.best_params['c_n'])

        # -------------------------------------------------------------
        # 🔧 第二轮精修：Refine C (固定质量与接触)
        # -------------------------------------------------------------
        print("\n🔁 [Refine C] 精修 mu_lat, com_xyz (固定质量与接触)...")
        def objective_C_refine(trial):
            p_dict = NOMINAL.copy()
            for key in ['mu_lat', 'com_dx', 'com_dy', 'com_dz']:
                p_dict[key] = normalizer.norm_to_phys(key, trial.suggest_float(key, 0, 1))
            sim_C = sim.simulate_stage_C(p_dict)
            com_xyz = (p_dict['com_dx'], p_dict['com_dy'], p_dict['com_dz'])
            return compute_loss_stage_C(sim_C, real_C_noisy, scales_C, com_xyz=com_xyz)

        study_C_ref = optuna.create_study(sampler=make_sampler(30), direction="minimize")
        study_C_ref.optimize(objective_C_refine, n_trials=200, show_progress_bar=False)
        loss_history['Refine C'] = study_C_ref.trials_dataframe()['value'].tolist()
        for key in ['mu_lat', 'com_dx', 'com_dy', 'com_dz']:
            NOMINAL[key] = normalizer.norm_to_phys(key, study_C_ref.best_params[key])

        # -------------------------------------------------------------
        # 🔧 Final Joint：6 维联合优化（sys_delay / joint_damp / mass / mu_spin 固定）
        # -------------------------------------------------------------
        print("\n🧩 [Final Joint] 6 维联合优化 (sys_delay / joint_damp / mass / mu_spin 固定)...")
        def objective_final(trial):
            p_dict = NOMINAL.copy()
            for key in ['mu_lat', 'k_n', 'c_n',
                        'com_dx', 'com_dy', 'com_dz']:
                p_dict[key] = normalizer.norm_to_phys(key, trial.suggest_float(key, 0, 1))
            sim_A = sim.simulate_stage_A(p_dict)
            sim_B = sim.simulate_stage_B(p_dict)
            sim_C = sim.simulate_stage_C(p_dict)
            loss_A = compute_loss_stage_A(sim_A, real_A_noisy, scales_A)
            loss_B = compute_loss_stage_B(sim_B, real_B_noisy, scales_B)
            com_xyz = (p_dict['com_dx'], p_dict['com_dy'], p_dict['com_dz'])
            loss_C = compute_loss_stage_C(sim_C, real_C_noisy, scales_C, com_xyz=com_xyz)
            return float(loss_A + loss_B + loss_C)

        study_final = optuna.create_study(sampler=make_sampler(40), direction="minimize")
        study_final.optimize(objective_final, n_trials=400, show_progress_bar=False)
        loss_history['Final Joint'] = study_final.trials_dataframe()['value'].tolist()
        for key in ['mu_lat', 'k_n', 'c_n',
                    'com_dx', 'com_dy', 'com_dz']:
            NOMINAL[key] = normalizer.norm_to_phys(key, study_final.best_params[key])

        # -------------------------------------------------------------
        # 最终结果
        # -------------------------------------------------------------
        final_calibrated_params = NOMINAL.copy()

        print("\n" + "=" * 70)
        print("🏆 [最终标定评估结果] 10 维参数标定误差对比表")
        print(f"{'Parameter':<14} | {'Ground Truth':<14} | {'Estimated':<14} | {'Error %':<10}")
        print("-" * 70)
        for key in TRUE_PARAMS:
            gt = TRUE_PARAMS[key]
            est = final_calibrated_params[key]
            if abs(gt) < 1e-6:
                err_str = f"{abs(est - gt):.5f} (abs)"
            else:
                err_pct = abs(est - gt) / abs(gt) * 100.0
                err_str = f"{err_pct:.2f}%"
            print(f"{key:<14} | {gt:<14.4f} | {est:<14.4f} | {err_str:<10}")
        print("=" * 70)

        # -------------------------------------------------------------
        # 干净数据最终评估（无噪声真值，衡量真实标定质量）
        # -------------------------------------------------------------
        print("\n🧪 [Clean-data 最终评估] 用无噪声真值数据评估标定结果...")
        scales_A_clean = {'q': np.var(clean_data['A'][0]) + 1e-5,
                          'v': np.var(clean_data['A'][1]) + 1e-5,
                          't': np.var(clean_data['A'][2]) + 1e-5}
        scales_B_clean = {'z': np.var(clean_data['B'][0]) + 1e-5,
                          'nf': np.var(clean_data['B'][1]) + 1e-5,
                          'tq': np.var(clean_data['B'][2]) + 1e-5}
        scales_C_clean = {'pos': np.var(clean_data['C'][0]) + 1e-5,
                          'orn': np.var(clean_data['C'][1]) + 1e-5,
                          'cf': np.var(clean_data['C'][2]) + 1e-5}
        sim_A_eval = sim.simulate_stage_A(final_calibrated_params)
        sim_B_eval = sim.simulate_stage_B(final_calibrated_params)
        sim_C_eval = sim.simulate_stage_C(final_calibrated_params)
        sim_S_eval = sim.simulate_stage_S(final_calibrated_params)
        clean_loss_A = compute_loss_stage_A(sim_A_eval, clean_data['A'], scales_A_clean)
        clean_loss_B = compute_loss_stage_B(sim_B_eval, clean_data['B'], scales_B_clean)
        com_xyz = (final_calibrated_params['com_dx'],
                   final_calibrated_params['com_dy'],
                   final_calibrated_params['com_dz'])
        clean_loss_C = compute_loss_stage_C(sim_C_eval, clean_data['C'], scales_C_clean, com_xyz=com_xyz)
        clean_loss_S = compute_loss_stage_S(sim_S_eval, clean_data['S'], np.var(clean_data['S']) + 1e-5)
        print(f"  干净数据损失  Stage A={clean_loss_A:.4f}  Stage B={clean_loss_B:.4f}  Stage C={clean_loss_C:.4f}  Stage S={clean_loss_S:.4f}")
        print(f"  干净数据总损失 = {clean_loss_A + clean_loss_B + clean_loss_C + clean_loss_S:.4f}")

        # 绘图：显示所有阶段 loss 曲线
        fig, axes = plt.subplots(3, 3, figsize=(20, 14))
        axes = axes.flatten()
        stage_titles = ['Stage A', 'Stage M', 'Stage B', 'Stage S', 'Stage C', 'Refine B', 'Refine C', 'Final Joint', 'All Stages']
        keys = ['Stage A', 'Stage M', 'Stage B', 'Stage S', 'Stage C', 'Refine B', 'Refine C', 'Final Joint']
        colors = ['#1f77b4', '#17becf', '#ff7f0e', '#e377c2', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for idx, key in enumerate(keys):
            ax = axes[idx]
            losses = loss_history[key]
            ax.plot(losses, color=colors[idx], lw=1.5, label=key)
            ax.set_title(stage_titles[idx], fontsize=12, fontweight='bold')
            ax.set_xlabel("Iteration", fontsize=10)
            ax.set_ylabel("Loss (Log Scale)", fontsize=10)
            ax.set_yscale("log")
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            ax.legend()
        # 最后一个子图：所有曲线
        ax = axes[8]
        for key, color in zip(keys, colors):
            ax.plot(loss_history[key], color=color, lw=1.5, label=key)
        ax.set_title("All Stages", fontsize=12, fontweight='bold')
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Loss (Log Scale)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig("loss_curves_final.png", dpi=300)
        plt.show()
        print("📊 Loss 曲线已保存至 'loss_curves_final.png'")

    finally:
        sim.disconnect()


if __name__ == "__main__":
    main()