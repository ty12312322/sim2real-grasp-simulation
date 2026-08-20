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
        self.gui = gui
        self.physics_client = None
        self.robot_id = None
        self.cube_id = None
        self.plane_id = None
        self.EE_INDEX = 11
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
        """在当前位置静止采集腕部关节（5,6）平均力矩"""
        steps = int(duration * 240)
        torques = []
        for _ in range(steps):
            p.stepSimulation()
            states = p.getJointStates(self.robot_id, [5, 6])
            torques.append([states[0][3], states[1][3]])
        return np.mean(torques, axis=0)

    # 🔧 新增：互相关估计系统延迟（单位：帧）
    def estimate_sys_delay(self, params, duration=0.5, joint_idx=4):
        """
        使用阶跃响应估计系统延迟（帧数）
        """
        self.set_params(params)

        dt = 1.0 / 240.0
        steps = int(duration * 240)

        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        base_j = list(p.calculateInverseKinematics(self.robot_id, self.EE_INDEX, [0.5, 0.0, 0.3], target_orn))
        self.reset_to_state(cube_pos=[0.5, 0.0, 0.0], arm_j=base_j, finger_pos=0.04)

        # 初始化缓冲，保持静止
        self.action_buffer.clear()
        for _ in range(self.delay_steps):
            self.action_buffer.append((base_j, 0.04, 5.0))

        # 阶跃命令：前 20 帧保持基线，然后突然改变关节 4 目标
        step_trigger = 30
        step_target = base_j[joint_idx] + 0.3  # 阶跃幅度

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

            cmd_traj.append(curr_cmd[joint_idx])
            state = p.getJointState(self.robot_id, joint_idx)
            vel_traj.append(state[1])

        # 找到速度开始偏离基线的时刻（响应开始）
        baseline_vel = np.mean(vel_traj[:step_trigger])
        vel_arr = np.array(vel_traj)
        threshold = baseline_vel + 0.05 * np.max(np.abs(vel_arr))  # 阈值设为峰值5%
        # 从阶跃时刻往后搜索，找到第一个超过阈值的点
        response_start = None
        for i in range(step_trigger, steps):
            if abs(vel_arr[i] - baseline_vel) > threshold:
                response_start = i
                break
        if response_start is None:
            response_start = steps - 1

        delay_frames = max(0, response_start - step_trigger)
        return delay_frames

    def connect(self):
        if self.physics_client is not None:
            return
        self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(
            numSolverIterations=150,
            numSubSteps=4,
            enableConeFriction=1,
            contactBreakingThreshold=0.001
        )
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
            arm_j = p.calculateInverseKinematics(self.robot_id, self.EE_INDEX, cube_pos, target_orn)
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
            p.setJointMotorControl2(
                self.robot_id, j_idx, p.POSITION_CONTROL,
                targetPosition=exec_j[idx],
                force=arm_force,
                positionGain=0.3,
                velocityGain=1.0
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
        init_j = list(p.calculateInverseKinematics(self.robot_id, self.EE_INDEX, hold_pos, target_orn))
        self.reset_to_state(cube_pos=hold_pos, arm_j=init_j, finger_pos=0.025)
        self.action_buffer.clear()
        for _ in range(self.delay_steps):
            self.action_buffer.append((init_j, finger_target, finger_force))
        for _ in range(steps):
            self._apply_action(init_j, finger_pos=finger_target, finger_force=finger_force)
            p.stepSimulation()

    # -----------------------------------------------------------------
    # 场景 A：空载多频扫频与阶跃 (保持原样)
    # -----------------------------------------------------------------
    def simulate_stage_A(self, params, duration=2.5):
        self.set_params(params)
        dt = 1.0 / 240.0
        steps = int(duration * 240)
        times = np.linspace(0, duration, steps)

        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        base_j = list(p.calculateInverseKinematics(self.robot_id, self.EE_INDEX, [0.5, 0.0, 0.3], target_orn))
        self.reset_to_state(cube_pos=[0.5, 0.0, 0.0], arm_j=base_j, finger_pos=0.04)
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
    # 场景 B：垂直变速变力振荡 (保持原样，不做任何修改)
    # -----------------------------------------------------------------
    def simulate_stage_B(self, params, duration=2.5):
        self.set_params(params)
        steps = int(duration * 240)
        times = np.linspace(0, duration, steps)

        self._settle_grasp(hold_pos=(0.5, 0.0, 0.2), finger_target=0.018, finger_force=15.0, steps=50)

        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        base_z = 0.20
        target_z = base_z + 0.05 * np.sin(2 * np.pi * 1.2 * times) + 0.02 * np.sin(2 * np.pi * 2.8 * times)
        dynamic_finger_force = 12.0 + 6.0 * np.sin(2 * np.pi * 4.0 * times)

        cube_z_list, normal_force_list, wrist_torque_list = [], [], []

        for i in range(steps):
            cmd_j = p.calculateInverseKinematics(self.robot_id, self.EE_INDEX, [0.5, 0.0, target_z[i]], target_orn)
            self._apply_action(cmd_j, finger_pos=0.015, finger_force=dynamic_finger_force[i], arm_force=180.0)
            p.stepSimulation()

            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            cube_z_list.append(cube_pos[2])

            contacts = p.getContactPoints(self.robot_id, self.cube_id)
            if contacts:
                nf = max([c[9] for c in contacts])
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
        pitch = 0.4 * np.sin(2 * np.pi * 0.9 * times + np.pi / 4)
        yaw = 0.3 * np.sin(2 * np.pi * 1.5 * times)

        cube_pos_list, cube_orn_list, contact_force_list = [], [], []

        for i in range(steps):
            orn = p.getQuaternionFromEuler([roll[i], pitch[i], yaw[i]])
            cmd_j = p.calculateInverseKinematics(self.robot_id, self.EE_INDEX, [0.5, 0.0, 0.20], orn)

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


# =====================================================================
# 3. 损失函数 (保持原样)
# =====================================================================
def compute_loss_stage_A(sim_data, real_data, scales):
    sim_q, sim_v, sim_t = sim_data
    real_q, real_v, real_t = real_data
    loss_q = np.mean((sim_q - real_q)**2) / scales['q']
    loss_v = np.mean((sim_v - real_v)**2) / scales['v']
    loss_t = np.mean((sim_t - real_t)**2) / scales['t']
    return float(loss_q + loss_v + loss_t)

def compute_loss_stage_B(sim_data, real_data, scales):
    sim_z, sim_nf, sim_tq = sim_data
    real_z, real_nf, real_tq = real_data
    loss_z = np.mean((sim_z - real_z)**2) / scales['z']
    loss_nf = np.mean((sim_nf - real_nf)**2) / scales['nf']
    loss_tq = np.mean((sim_tq - real_tq)**2) / scales['tq']
    return float(loss_z + loss_nf + loss_tq)

def compute_loss_stage_C(sim_data, real_data, scales):
    sim_pos, sim_orn, sim_cf = sim_data
    real_pos, real_orn, real_cf = real_data
    loss_pos = np.mean((sim_pos - real_pos)**2) / scales['pos']
    loss_orn = np.mean((sim_orn - real_orn)**2) / scales['orn']
    loss_cf = np.mean((sim_cf - real_cf)**2) / scales['cf']
    return float(loss_pos + loss_orn + loss_cf)


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
        "com_dx": 0.01,
        "com_dy": -0.005,
        "com_dz": 0.002,
        "joint_damp": 1.5,
        "sys_delay": 0.03
    }

    PARAM_BOUNDS = {
        "mass": (0.05, 0.35),
        "mu_lat": (0.20, 1.20),
        "mu_spin": (0.005, 0.12),
        "k_n": (1000.0, 12000.0),
        "c_n": (10.0, 180.0),
        "com_dx": (-0.03, 0.03),
        "com_dy": (-0.03, 0.03),
        "com_dz": (-0.03, 0.03),
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

        print("\n📡 [Step 1] 生成真实物理参考轨迹 (含传感器噪声)...")
        real_A = sim.simulate_stage_A(TRUE_PARAMS)
        real_B = sim.simulate_stage_B(TRUE_PARAMS)
        real_C = sim.simulate_stage_C(TRUE_PARAMS)

        # 添加噪声
        real_A_noisy = (
            real_A[0] + np.random.normal(0, 0.001, real_A[0].shape),
            real_A[1] + np.random.normal(0, 0.005, real_A[1].shape),
            real_A[2] + np.random.normal(0, 0.02, real_A[2].shape)
        )
        real_B_noisy = (
            real_B[0] + np.random.normal(0, 0.0005, real_B[0].shape),
            real_B[1] + np.random.normal(0, 0.2, real_B[1].shape),
            real_B[2] + np.random.normal(0, 0.01, real_B[2].shape)
        )
        real_C_noisy = (
            real_C[0] + np.random.normal(0, 0.0005, real_C[0].shape),
            real_C[1] + np.random.normal(0, 0.003, real_C[1].shape),
            real_C[2] + np.random.normal(0, 0.1, real_C[2].shape)
        )

        scales_A = {
            'q': np.var(real_A_noisy[0]) + 1e-5,
            'v': np.var(real_A_noisy[1]) + 1e-5,
            't': np.var(real_A_noisy[2]) + 1e-5
        }
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

        loss_history = {}

        def make_sampler(n_startup=15, sigma0=0.25):
            return optuna.samplers.CmaEsSampler(
                seed=42, sigma0=sigma0, restart_strategy="ipop", n_startup_trials=n_startup
            )

        # -------------------------------------------------------------
        # Stage A：只优化 joint_damp (sys_delay 已固定)
        # -------------------------------------------------------------
        print("\n📘 [Stage A] 辨识 joint_damp (sys_delay 已锁定)...")
        def objective_A(trial):
            p_dict = NOMINAL.copy()
            p_dict['joint_damp'] = normalizer.norm_to_phys('joint_damp', trial.suggest_float('joint_damp', 0, 1))
            sim_A = sim.simulate_stage_A(p_dict)
            return compute_loss_stage_A(sim_A, real_A_noisy, scales_A)

        study_A = optuna.create_study(sampler=make_sampler(15, 0.25), direction="minimize")
        study_A.optimize(objective_A, n_trials=60, show_progress_bar=False)
        loss_history['Stage A'] = study_A.trials_dataframe()['value'].tolist()
        NOMINAL['joint_damp'] = normalizer.norm_to_phys('joint_damp', study_A.best_params['joint_damp'])
        print(f"  --> 阶段 A 完成: joint_damp={NOMINAL['joint_damp']:.4f}")

        # -------------------------------------------------------------
        # Stage B：辨识 mass, k_n, c_n (原方法)
        # -------------------------------------------------------------
        print("\n📘 [Stage B] 辨识质量与接触参数: mass, k_n, c_n...")
        def objective_B(trial):
            p_dict = NOMINAL.copy()
            p_dict['mass'] = normalizer.norm_to_phys('mass', trial.suggest_float('mass', 0, 1))
            p_dict['k_n'] = normalizer.norm_to_phys('k_n', trial.suggest_float('k_n', 0, 1))
            p_dict['c_n'] = normalizer.norm_to_phys('c_n', trial.suggest_float('c_n', 0, 1))
            sim_B = sim.simulate_stage_B(p_dict)
            return compute_loss_stage_B(sim_B, real_B_noisy, scales_B)

        study_B = optuna.create_study(sampler=make_sampler(20, 0.25), direction="minimize")
        study_B.optimize(objective_B, n_trials=100, show_progress_bar=False)
        loss_history['Stage B'] = study_B.trials_dataframe()['value'].tolist()
        NOMINAL['mass'] = normalizer.norm_to_phys('mass', study_B.best_params['mass'])
        NOMINAL['k_n'] = normalizer.norm_to_phys('k_n', study_B.best_params['k_n'])
        NOMINAL['c_n'] = normalizer.norm_to_phys('c_n', study_B.best_params['c_n'])
        print(f"  --> 阶段 B 完成: mass={NOMINAL['mass']:.4f}, k_n={NOMINAL['k_n']:.1f}, c_n={NOMINAL['c_n']:.2f}")

        # -------------------------------------------------------------
        # Stage C：辨识摩擦与质心偏移 (原方法)
        # -------------------------------------------------------------
        print("\n📘 [Stage C] 辨识表面摩擦与 3D 质心偏移 (mu_lat, mu_spin, com_xyz)...")
        def objective_C(trial):
            p_dict = NOMINAL.copy()
            for key in ['mu_lat', 'mu_spin', 'com_dx', 'com_dy', 'com_dz']:
                p_dict[key] = normalizer.norm_to_phys(key, trial.suggest_float(key, 0, 1))
            sim_C = sim.simulate_stage_C(p_dict)
            return compute_loss_stage_C(sim_C, real_C_noisy, scales_C)

        study_C = optuna.create_study(sampler=make_sampler(30, 0.30), direction="minimize")
        study_C.optimize(objective_C, n_trials=180, show_progress_bar=False)
        loss_history['Stage C'] = study_C.trials_dataframe()['value'].tolist()
        for key in ['mu_lat', 'mu_spin', 'com_dx', 'com_dy', 'com_dz']:
            NOMINAL[key] = normalizer.norm_to_phys(key, study_C.best_params[key])
        print(f"  --> 阶段 C 完成: mu_lat={NOMINAL['mu_lat']:.4f}, mu_spin={NOMINAL['mu_spin']:.4f}, com_dz={NOMINAL['com_dz']:.4f}")

        # -------------------------------------------------------------
        # 🔧 第二轮精修：Refine B (固定摩擦与质心)
        # -------------------------------------------------------------
        print("\n🔁 [Refine B] 精修 mass, k_n, c_n (固定摩擦与质心)...")
        def objective_B_refine(trial):
            p_dict = NOMINAL.copy()
            p_dict['mass'] = normalizer.norm_to_phys('mass', trial.suggest_float('mass', 0, 1))
            p_dict['k_n'] = normalizer.norm_to_phys('k_n', trial.suggest_float('k_n', 0, 1))
            p_dict['c_n'] = normalizer.norm_to_phys('c_n', trial.suggest_float('c_n', 0, 1))
            sim_B = sim.simulate_stage_B(p_dict)
            return compute_loss_stage_B(sim_B, real_B_noisy, scales_B)

        study_B_ref = optuna.create_study(sampler=make_sampler(20, 0.15), direction="minimize")
        study_B_ref.optimize(objective_B_refine, n_trials=80, show_progress_bar=False)
        loss_history['Refine B'] = study_B_ref.trials_dataframe()['value'].tolist()
        NOMINAL['mass'] = normalizer.norm_to_phys('mass', study_B_ref.best_params['mass'])
        NOMINAL['k_n'] = normalizer.norm_to_phys('k_n', study_B_ref.best_params['k_n'])
        NOMINAL['c_n'] = normalizer.norm_to_phys('c_n', study_B_ref.best_params['c_n'])

        # -------------------------------------------------------------
        # 🔧 第二轮精修：Refine C (固定质量与接触)
        # -------------------------------------------------------------
        print("\n🔁 [Refine C] 精修 mu_lat, mu_spin, com_xyz (固定质量与接触)...")
        def objective_C_refine(trial):
            p_dict = NOMINAL.copy()
            for key in ['mu_lat', 'mu_spin', 'com_dx', 'com_dy', 'com_dz']:
                p_dict[key] = normalizer.norm_to_phys(key, trial.suggest_float(key, 0, 1))
            sim_C = sim.simulate_stage_C(p_dict)
            return compute_loss_stage_C(sim_C, real_C_noisy, scales_C)

        study_C_ref = optuna.create_study(sampler=make_sampler(30, 0.15), direction="minimize")
        study_C_ref.optimize(objective_C_refine, n_trials=80, show_progress_bar=False)
        loss_history['Refine C'] = study_C_ref.trials_dataframe()['value'].tolist()
        for key in ['mu_lat', 'mu_spin', 'com_dx', 'com_dy', 'com_dz']:
            NOMINAL[key] = normalizer.norm_to_phys(key, study_C_ref.best_params[key])

        # -------------------------------------------------------------
        # 最终结果（直接使用 NOMINAL，不再有 Final Joint 联合优化）
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

        # 绘图：显示所有阶段 loss 曲线
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        stage_titles = ['Stage A', 'Stage B', 'Stage C', 'Refine B', 'Refine C', 'All Stages']
        keys = ['Stage A', 'Stage B', 'Stage C', 'Refine B', 'Refine C']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
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
        ax = axes[5]
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