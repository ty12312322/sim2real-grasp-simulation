import time
import numpy as np
import pybullet as p
import pybullet_data
import optuna
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import butter, filtfilt

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 物理仿真器（重构版）
# =====================================================================
class RobotSimulator:
    def __init__(self, gui=False):
        self.gui = gui
        self.physics_client = None
        self.robot_id = None
        self.cube_id = None
        self.EE_INDEX = 11
        self.FINGER_L, self.FINGER_R = 9, 10
        self.use_external_torque_com = False

    @staticmethod
    def _to_float(val):
        """安全转换为 float，处理可能出现的元组/列表类型"""
        if isinstance(val, (int, float)):
            return float(val)
        elif isinstance(val, (tuple, list, np.ndarray)):
            if len(val) == 1:
                return float(val[0])
            else:
                # 如果是一个向量，返回其模长
                return float(np.linalg.norm(val))
        else:
            return 0.0

    def connect(self):
        self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSolverIterations=150, numSubSteps=10, enableConeFriction=1)
        p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0.0, 0.025])

    def disconnect(self):
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
    def set_params(self, params):
        # 摩擦与接触参数
        p.changeDynamics(self.cube_id, -1,
                        mass=params['mass'],
                        lateralFriction=params['mu_lat'],
                        spinningFriction=params['mu_spin'],
                        contactStiffness=params['k_n'],
                        contactDamping=params['c_n'])

        # 手指摩擦
        p.changeDynamics(self.robot_id, self.FINGER_L, lateralFriction=params['mu_lat'])
        p.changeDynamics(self.robot_id, self.FINGER_R, lateralFriction=params['mu_lat'])

        # 关节阻尼
        for i in range(7):
            p.changeDynamics(self.robot_id, i, jointDamping=params['joint_damp'])
            p.resetJointState(self.robot_id, i, 0.0)

        # 尝试使用 localInertialFramePosition 设置质心偏移（新版本支持）
        try:
            p.changeDynamics(self.cube_id, -1,
                            localInertialFramePosition=[params['com_dx'], params['com_dy'], params['com_dz']],
                            localInertialFrameOrientation=[0, 0, 0, 1])
            self.use_external_torque_com = False
            print("使用 localInertialFramePosition 设置质心偏移")
        except TypeError:
            # 回退：使用外部重力矩模拟质心偏移（精度稍低但可运行）
            print("警告：当前 PyBullet 版本不支持 localInertialFramePosition，回退到外部力矩方法（精度可能下降）")
            self.use_external_torque_com = True

        # 系统延迟缓存
        self.delay_steps = max(1, int(params['sys_delay'] * 240))
        self.action_buffer = deque(maxlen=self.delay_steps)
        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        initial_j = p.calculateInverseKinematics(self.robot_id, self.EE_INDEX,
                                                [0.5, 0.0, 0.025], target_orn)
        for _ in range(self.delay_steps):
            self.action_buffer.append((initial_j, 5.0, target_orn))

    def _apply_action(self, target_j, force, target_orn, finger_pos, finger_force):
        """将动作送入延迟队列，并执行队列最前端的动作"""
        self.action_buffer.append((target_j, finger_force, target_orn))
        exec_j, exec_force, exec_orn = self.action_buffer[0]

        for i in range(7):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                    targetPosition=exec_j[i], force=200)
        # 夹爪控制
        p.setJointMotorControl2(self.robot_id, self.FINGER_L, p.POSITION_CONTROL,
                                targetPosition=finger_pos, force=exec_force)
        p.setJointMotorControl2(self.robot_id, self.FINGER_R, p.POSITION_CONTROL,
                                targetPosition=finger_pos, force=exec_force)

    # -----------------------------------------------------------------
    # 场景 A：空载扫频（用于辨识关节阻尼和系统延迟）
    # -----------------------------------------------------------------
    def simulate_stage_A(self, params, duration=3.0, freq_sweep=(0.2, 3.0)):
        self.set_params(params)
        dt = 1.0 / 240.0
        steps = int(duration * 240)
        times = np.linspace(0, duration, steps)

        # 生成扫频参考轨迹（正弦扫频）
        freq = np.linspace(freq_sweep[0], freq_sweep[1], steps)
        phase = 2 * np.pi * np.cumsum(freq) * dt
        target_angle = 0.3 * np.sin(phase)  # 一个关节的正弦运动

        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        joint_traj = []
        vel_traj = []
        torque_traj = []

        for i in range(steps):
            # 只驱动第 4 关节（肩关节）作为激励，其他关节保持固定
            base_j = list(p.calculateInverseKinematics(self.robot_id, self.EE_INDEX,
                                           [0.5, 0.0, 0.3], target_orn))
            base_j[4] = target_angle[i]  # 修改关节 4
            self._apply_action(base_j, 0.0, target_orn, finger_pos=0.04, finger_force=5.0)
            p.stepSimulation()

            states = p.getJointStates(self.robot_id, range(7))
            joint_traj.append([s[0] for s in states])
            vel_traj.append([s[1] for s in states])
            torque_traj.append([s[2][3:6] for s in states])  # 关节反作用力矩（Mx, My, Mz）
        return (np.array(joint_traj), np.array(vel_traj), np.array(torque_traj))

    # -----------------------------------------------------------------
    # 场景 B：夹持垂直运动（用于辨识质量、接触刚度、阻尼）
    # -----------------------------------------------------------------
    def simulate_stage_B(self, params, duration=2.5, freq=0.8, amplitude=0.04):
        self.set_params(params)
        dt = 1.0 / 240.0
        steps = int(duration * 240)
        times = np.linspace(0, duration, steps)

        # 垂直正弦运动（低频）
        base_height = 0.15
        target_z = base_height + amplitude * np.sin(2 * np.pi * freq * times)

        target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        z_traj = []
        acc_traj = []
        normal_force_traj = []
        lateral_force_traj = []
        joint_torque_traj = []

        for i in range(steps):
            cmd_j = p.calculateInverseKinematics(self.robot_id, self.EE_INDEX,
                                                 [0.5, 0.0, target_z[i]], target_orn)
            self._apply_action(cmd_j, 0.0, target_orn, finger_pos=0.02, finger_force=15.0)
            p.stepSimulation()

            cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
            cube_vel, cube_ang_vel = p.getBaseVelocity(self.cube_id)
            z_traj.append(cube_pos[2])
            acc_traj.append(cube_vel[2])  # 速度差分可得加速度，但为了简单用速度

            # 接触力
            contacts = p.getContactPoints(self.robot_id, self.cube_id)
            if contacts:
                normal = self._to_float(max([c[9] for c in contacts]))
                lat1 = self._to_float(max([c[10] for c in contacts]))
                lat2 = self._to_float(max([c[11] for c in contacts]))
                normal_force_traj.append(normal)
                lateral_force_traj.append(np.sqrt(lat1**2 + lat2**2))
            else:
                normal_force_traj.append(0.0)
                lateral_force_traj.append(0.0)

            # 关节力矩（第 6、7 关节，靠近腕部）
            states = p.getJointStates(self.robot_id, [5, 6])
            joint_torque_traj.append([states[0][2][3], states[1][2][3]])  # 取关节5、6的Mx力矩

        return (np.array(z_traj), np.array(acc_traj),
                np.array(normal_force_traj), np.array(lateral_force_traj),
                np.array(joint_torque_traj))

    # -----------------------------------------------------------------
    # 场景 C：夹持旋转滑动（用于辨识摩擦、质心偏移）
    # -----------------------------------------------------------------

    def simulate_stage_C(self, params, duration=3.0):
        self.set_params(params)
        dt = 1.0 / 240.0
        steps = int(duration * 240)
        times = np.linspace(0, duration, steps)

        # 初始夹持水平，然后缓慢旋转腕部使 cube 翻转
        target_orn_initial = p.getQuaternionFromEuler([np.pi, 0, 0])
        z_const = 0.1
        roll_angle = np.pi + 0.8 * np.sin(2 * np.pi * 0.2 * times)  # 缓慢摇晃

        cube_pos_traj = []
        cube_orn_traj = []
        cube_ang_vel_traj = []
        contact_force_traj = []

        for i in range(steps):
            orn = p.getQuaternionFromEuler([roll_angle[i], 0, 0])
            cmd_j = p.calculateInverseKinematics(self.robot_id, self.EE_INDEX,
                                                [0.5, 0.0, z_const], orn)
            self._apply_action(cmd_j, 0.0, orn, finger_pos=0.022, finger_force=8.0)

            # 如果使用外部力矩回退，施加重力矩
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
            vel, ang_vel = p.getBaseVelocity(self.cube_id)
            # 强制转换为纯 float 列表
            cube_pos_traj.append([float(pos[0]), float(pos[1]), float(pos[2])])
            cube_orn_traj.append(list(map(float, p.getEulerFromQuaternion(orn_cube))))
            cube_ang_vel_traj.append([float(ang_vel[0]), float(ang_vel[1]), float(ang_vel[2])])

            contacts = p.getContactPoints(self.robot_id, self.cube_id)
            contacts = p.getContactPoints(self.robot_id, self.cube_id)
            if contacts:
                normal = self._to_float(max([c[9] for c in contacts]))
                lat1 = self._to_float(max([c[10] for c in contacts]))
                lat2 = self._to_float(max([c[11] for c in contacts]))
                contact_force_traj.append([normal, lat1, lat2])
            else:
                contact_force_traj.append([0.0, 0.0, 0.0])

        return (np.array(cube_pos_traj), np.array(cube_orn_traj),
                np.array(cube_ang_vel_traj), np.array(contact_force_traj))

# =====================================================================
# 参数空间归一化与对数尺度处理
# =====================================================================
class ParamNormalizer:
    def __init__(self, bounds):
        self.bounds = bounds
        self.log_params = {'k_n', 'c_n'}  # 对这两个参数使用对数尺度

    def transform(self, x_norm):
        """将 [0,1]^n 映射到实际参数值"""
        params = {}
        idx = 0
        for key, (low, high) in self.bounds.items():
            v = x_norm[idx]
            if key in self.log_params:
                # 对数均匀
                log_low = np.log(low)
                log_high = np.log(high)
                params[key] = np.exp(log_low + v * (log_high - log_low))
            else:
                params[key] = low + v * (high - low)
            idx += 1
        return params

    def inverse_transform(self, params):
        """实际参数 -> [0,1]"""
        x = []
        for key, (low, high) in self.bounds.items():
            val = params[key]
            if key in self.log_params:
                log_low = np.log(low)
                log_high = np.log(high)
                x.append((np.log(val) - log_low) / (log_high - log_low))
            else:
                x.append((val - low) / (high - low))
        return np.array(x)

    def get_bounds_list(self):
        return list(self.bounds.values())


# =====================================================================
# 目标函数构建（多阶段）
# =====================================================================
def compute_loss_stage_A(sim_data, real_data, scales):
    sim_q, sim_v, sim_t = sim_data
    real_q, real_v, real_t = real_data
    loss_q = np.mean((sim_q - real_q)**2, axis=0).sum() / scales['q']
    loss_v = np.mean((sim_v - real_v)**2, axis=0).sum() / scales['v']
    loss_t = np.mean((sim_t - real_t)**2, axis=0).sum() / scales['t']
    return loss_q + loss_v + loss_t

def compute_loss_stage_B(sim_data, real_data, scales):
    sim_z, sim_acc, sim_nf, sim_lf, sim_tq = sim_data
    real_z, real_acc, real_nf, real_lf, real_tq = real_data
    loss_z = np.mean((sim_z - real_z)**2) / scales['z']
    loss_acc = np.mean((sim_acc - real_acc)**2) / scales['acc']
    loss_nf = np.mean((sim_nf - real_nf)**2) / scales['nf']
    loss_lf = np.mean((sim_lf - real_lf)**2) / scales['lf']
    loss_tq = np.mean((sim_tq - real_tq)**2) / scales['tq']
    return loss_z + loss_acc + loss_nf + loss_lf + loss_tq

def compute_loss_stage_C(sim_data, real_data, scales):
    sim_pos, sim_orn, sim_av, sim_cf = sim_data
    real_pos, real_orn, real_av, real_cf = real_data
    loss_pos = np.mean((sim_pos - real_pos)**2) / scales['pos']
    loss_orn = np.mean((sim_orn - real_orn)**2) / scales['orn']
    loss_av = np.mean((sim_av - real_av)**2) / scales['av']
    loss_cf = np.mean((sim_cf - real_cf)**2) / scales['cf']
    return loss_pos + loss_orn + loss_av + loss_cf


# =====================================================================
# 主流程
# =====================================================================
def main():
    np.random.seed(42)
    print("="*60)
    print("🔧 工业级物理参数标定（改进版）")
    print("="*60)

    # -----------------------------------------------------------------
    # 1. 定义真实参数（上帝视角）与搜索边界
    # -----------------------------------------------------------------
    TRUE_PARAMS = {
        "mass": 0.15, "mu_lat": 0.80, "mu_spin": 0.05,
        "k_n": 5000.0, "c_n": 50.0,
        "com_dx": 0.01, "com_dy": -0.005, "com_dz": 0.002,
        "joint_damp": 1.5, "sys_delay": 0.03
    }

    PARAM_BOUNDS = {
        "mass": (0.05, 0.3),
        "mu_lat": (0.2, 1.2),
        "mu_spin": (0.0, 0.1),
        "k_n": (1000.0, 10000.0),
        "c_n": (10.0, 150.0),
        "com_dx": (-0.03, 0.03),
        "com_dy": (-0.03, 0.03),
        "com_dz": (-0.03, 0.03),
        "joint_damp": (0.1, 3.0),
        "sys_delay": (0.0, 0.08)
    }

    # 初始猜测（课程起点）
    NOMINAL = {
        "mass": 0.05, "mu_lat": 0.20, "mu_spin": 0.01,
        "k_n": 1000.0, "c_n": 10.0,
        "com_dx": 0.0, "com_dy": 0.0, "com_dz": 0.0,
        "joint_damp": 0.1, "sys_delay": 0.0
    }

    normalizer = ParamNormalizer(PARAM_BOUNDS)

    # -----------------------------------------------------------------
    # 2. 生成真实世界数据（含噪声）
    # -----------------------------------------------------------------
    print("📡 生成真实数据...")
    sim = RobotSimulator(gui=False)
    sim.connect()
    try:
        # 场景 A
        real_A = sim.simulate_stage_A(TRUE_PARAMS)
        # 场景 B
        real_B = sim.simulate_stage_B(TRUE_PARAMS)
        # 场景 C
        real_C = sim.simulate_stage_C(TRUE_PARAMS)
    finally:
        sim.disconnect()

    # 添加噪声
    noise = {
        'A': {'q': 0.001, 'v': 0.005, 't': 0.01},
        'B': {'z': 0.0005, 'acc': 0.002, 'nf': 0.5, 'lf': 0.3, 'tq': 0.02},
        'C': {'pos': 0.0005, 'orn': 0.005, 'av': 0.01, 'cf': 0.2}
    }
    real_A_noisy = (
        real_A[0] + np.random.normal(0, noise['A']['q'], real_A[0].shape),
        real_A[1] + np.random.normal(0, noise['A']['v'], real_A[1].shape),
        real_A[2] + np.random.normal(0, noise['A']['t'], real_A[2].shape)
    )
    real_B_noisy = (
        real_B[0] + np.random.normal(0, noise['B']['z'], real_B[0].shape),
        real_B[1] + np.random.normal(0, noise['B']['acc'], real_B[1].shape),
        real_B[2] + np.random.normal(0, noise['B']['nf'], real_B[2].shape),
        real_B[3] + np.random.normal(0, noise['B']['lf'], real_B[3].shape),
        real_B[4] + np.random.normal(0, noise['B']['tq'], real_B[4].shape)
    )
    real_C_noisy = (
        real_C[0] + np.random.normal(0, noise['C']['pos'], real_C[0].shape),
        real_C[1] + np.random.normal(0, noise['C']['orn'], real_C[1].shape),
        real_C[2] + np.random.normal(0, noise['C']['av'], real_C[2].shape),
        real_C[3] + np.random.normal(0, noise['C']['cf'], real_C[3].shape)
    )

    # 计算归一化尺度（避免 loss 被大方差主导）
    scales_A = {
        'q': np.var(real_A_noisy[0]) + 1e-6,
        'v': np.var(real_A_noisy[1]) + 1e-6,
        't': np.var(real_A_noisy[2]) + 1e-6
    }
    scales_B = {
        'z': np.var(real_B_noisy[0]) + 1e-6,
        'acc': np.var(real_B_noisy[1]) + 1e-6,
        'nf': np.var(real_B_noisy[2]) + 1e-6,
        'lf': np.var(real_B_noisy[3]) + 1e-6,
        'tq': np.var(real_B_noisy[4]) + 1e-6
    }
    scales_C = {
        'pos': np.var(real_C_noisy[0]) + 1e-6,
        'orn': np.var(real_C_noisy[1]) + 1e-6,
        'av': np.var(real_C_noisy[2]) + 1e-6,
        'cf': np.var(real_C_noisy[3]) + 1e-6
    }

    # -----------------------------------------------------------------
    # 3. 课程优化
    # -----------------------------------------------------------------
    loss_history = {}

    def create_study(n_trials, n_startup_trials=10):
        sampler = optuna.samplers.CmaEsSampler(
            seed=42,
            sigma0=0.3,
            restart_strategy="ipop",
            n_startup_trials=n_startup_trials
        )
        return optuna.create_study(sampler=sampler, direction="minimize")

    # ---------- 课程 1：关节阻尼 + 延迟 ----------
    print("\n📘 [Stage A] 辨识 joint_damp 和 sys_delay...")
    def objective_A(trial):
        x = np.array([
            trial.suggest_float("joint_damp", 0, 1),
            trial.suggest_float("sys_delay", 0, 1)
        ])
        # 其他参数使用名义值
        params = NOMINAL.copy()
        params["joint_damp"] = PARAM_BOUNDS["joint_damp"][0] + x[0] * (PARAM_BOUNDS["joint_damp"][1] - PARAM_BOUNDS["joint_damp"][0])
        params["sys_delay"] = PARAM_BOUNDS["sys_delay"][0] + x[1] * (PARAM_BOUNDS["sys_delay"][1] - PARAM_BOUNDS["sys_delay"][0])

        sim = RobotSimulator(gui=False)
        sim.connect()
        try:
            sim_A = sim.simulate_stage_A(params)
        finally:
            sim.disconnect()
        loss = compute_loss_stage_A(sim_A, real_A_noisy, scales_A)
        return loss

    study_A = create_study(80, n_startup_trials=20)
    study_A.optimize(objective_A, n_trials=80, show_progress_bar=False)
    loss_history['Stage A'] = study_A.trials_dataframe()['value'].tolist()

    # 更新名义值
    NOMINAL["joint_damp"] = study_A.best_params["joint_damp"]
    NOMINAL["sys_delay"] = study_A.best_params["sys_delay"]

    # ---------- 课程 2：质量、刚度、阻尼 ----------
    print("\n📘 [Stage B] 辨识 mass, k_n, c_n...")
    def objective_B(trial):
        x = np.array([
            trial.suggest_float("mass", 0, 1),
            trial.suggest_float("k_n", 0, 1),
            trial.suggest_float("c_n", 0, 1)
        ])
        params = NOMINAL.copy()
        # 对数尺度
        log_low, log_high = np.log(PARAM_BOUNDS["k_n"][0]), np.log(PARAM_BOUNDS["k_n"][1])
        params["k_n"] = np.exp(log_low + x[1] * (log_high - log_low))
        log_low, log_high = np.log(PARAM_BOUNDS["c_n"][0]), np.log(PARAM_BOUNDS["c_n"][1])
        params["c_n"] = np.exp(log_low + x[2] * (log_high - log_low))
        params["mass"] = PARAM_BOUNDS["mass"][0] + x[0] * (PARAM_BOUNDS["mass"][1] - PARAM_BOUNDS["mass"][0])

        sim = RobotSimulator(gui=False)
        sim.connect()
        try:
            sim_B = sim.simulate_stage_B(params)
        finally:
            sim.disconnect()
        loss = compute_loss_stage_B(sim_B, real_B_noisy, scales_B)
        return loss

    study_B = create_study(120, n_startup_trials=30)
    study_B.optimize(objective_B, n_trials=120, show_progress_bar=False)
    loss_history['Stage B'] = study_B.trials_dataframe()['value'].tolist()

    NOMINAL["mass"] = study_B.best_params["mass"]
    NOMINAL["k_n"] = study_B.best_params["k_n"]
    NOMINAL["c_n"] = study_B.best_params["c_n"]

    # ---------- 课程 3：摩擦、质心偏移 ----------
    print("\n📘 [Stage C] 辨识 mu_lat, mu_spin, com_dx/dy/dz...")
    def objective_C(trial):
        x = np.array([
            trial.suggest_float("mu_lat", 0, 1),
            trial.suggest_float("mu_spin", 0, 1),
            trial.suggest_float("com_dx", 0, 1),
            trial.suggest_float("com_dy", 0, 1),
            trial.suggest_float("com_dz", 0, 1)
        ])
        params = NOMINAL.copy()
        bounds_list = [PARAM_BOUNDS[k] for k in ["mu_lat", "mu_spin", "com_dx", "com_dy", "com_dz"]]
        for idx, key in enumerate(["mu_lat", "mu_spin", "com_dx", "com_dy", "com_dz"]):
            low, high = bounds_list[idx]
            params[key] = low + x[idx] * (high - low)

        sim = RobotSimulator(gui=False)
        sim.connect()
        try:
            sim_C = sim.simulate_stage_C(params)
        finally:
            sim.disconnect()
        loss = compute_loss_stage_C(sim_C, real_C_noisy, scales_C)
        return loss

    study_C = create_study(160, n_startup_trials=40)
    study_C.optimize(objective_C, n_trials=160, show_progress_bar=False)
    loss_history['Stage C'] = study_C.trials_dataframe()['value'].tolist()

    NOMINAL.update({
        "mu_lat": study_C.best_params["mu_lat"],
        "mu_spin": study_C.best_params["mu_spin"],
        "com_dx": study_C.best_params["com_dx"],
        "com_dy": study_C.best_params["com_dy"],
        "com_dz": study_C.best_params["com_dz"]
    })

    # -----------------------------------------------------------------
    # 4. 最终联合优化（全 10 维）
    # -----------------------------------------------------------------
    print("\n🚀 [Final Joint Optimization] 联合优化所有 10 个参数...")
    def objective_final(trial):
        x = np.array([trial.suggest_float(f"p{i}", 0, 1) for i in range(10)])
        params = normalizer.transform(x)

        sim = RobotSimulator(gui=False)
        sim.connect()
        try:
            sim_A = sim.simulate_stage_A(params)
            sim_B = sim.simulate_stage_B(params)
            sim_C = sim.simulate_stage_C(params)
        finally:
            sim.disconnect()

        loss_A = compute_loss_stage_A(sim_A, real_A_noisy, scales_A)
        loss_B = compute_loss_stage_B(sim_B, real_B_noisy, scales_B)
        loss_C = compute_loss_stage_C(sim_C, real_C_noisy, scales_C)
        # 权重可调
        total_loss = loss_A + loss_B + loss_C
        return total_loss

    study_final = create_study(300, n_startup_trials=80)
    study_final.optimize(objective_final, n_trials=300, show_progress_bar=False)
    loss_history['Final Joint'] = study_final.trials_dataframe()['value'].tolist()

    # 提取最终参数
    final_x = np.array([study_final.best_params[f"p{i}"] for i in range(10)])
    final_params = normalizer.transform(final_x)

    # -----------------------------------------------------------------
    # 5. 结果输出与可视化
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print("🏆 [最终结果] 10 维参数标定误差")
    print(f"{'Parameter':<15} | {'Ground Truth':<15} | {'Estimated':<15} | {'Error %':<10}")
    print("-" * 60)
    for key in TRUE_PARAMS:
        true_val = TRUE_PARAMS[key]
        est_val = final_params[key]
        if abs(true_val) < 1e-5:
            err_str = "N/A"
        else:
            err_str = f"{abs(est_val - true_val) / abs(true_val) * 100:.1f}%"
        print(f"{key:<15} | {true_val:<15.4f} | {est_val:<15.4f} | {err_str:<10}")
    print("="*60)

    # 绘制 loss 曲线
    plt.figure(figsize=(12, 6))
    for stage, losses in loss_history.items():
        plt.plot(losses, label=stage)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.title("Loss Curves per Optimization Stage")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("loss_curves.png")
    plt.show()
    print("\n📊 Loss 曲线已保存为 'loss_curves.png'")


if __name__ == "__main__":
    main()