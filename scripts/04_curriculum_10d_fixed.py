import time
import numpy as np
import pybullet as p
import pybullet_data
import optuna
import matplotlib.pyplot as plt
from collections import deque

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 🛠️ 核心引擎：支持多场景切换的 10 维黑盒仿真器 (全漏洞修复版)
# =====================================================================
def simulate_stage(stage, params, gui=False):
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    p.setPhysicsEngineParameter(numSolverIterations=150, numSubSteps=10, enableConeFriction=1)

    p.loadURDF("plane.urdf")
    cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0.0, 0.025])
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    EE_INDEX = 11  
    FINGER_L, FINGER_R = 9, 10   

    # 1. 注入 10 维物理参数
    p.changeDynamics(cube_id, -1, 
                     mass=params['mass'], 
                     lateralFriction=params['mu_lat'],
                     spinningFriction=params['mu_spin'], 
                     contactStiffness=params['k_n'], 
                     contactDamping=params['c_n']) 

    p.changeDynamics(robot_id, FINGER_L, lateralFriction=params['mu_lat'])
    p.changeDynamics(robot_id, FINGER_R, lateralFriction=params['mu_lat'])
    for i in range(7):
        p.changeDynamics(robot_id, i, jointDamping=params['joint_damp'])
        p.resetJointState(robot_id, i, 0.0)

    # 2. 初始化系统延迟 Buffer (修复时间穿梭Bug，强制所有动作过队列)
    delay_steps = max(1, int(params['sys_delay'] * 240))
    action_buffer = deque(maxlen=delay_steps)
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0]) 
    initial_j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, 0.0, 0.025], target_orn)
    for _ in range(delay_steps): 
        action_buffer.append((initial_j, 5.0, target_orn))

    traj_z, traj_f, traj_qvel, traj_roll = [], [], [], []

    try:
        # ==============================================================
        # 🎬 场景 1：空中乱舞 (专攻阻尼、延迟)
        # ==============================================================
        if stage == 1:
            for step in range(150):
                y_sweep = 0.2 * np.sin(step * 0.2)
                z_sweep = 0.3 + 0.1 * np.cos(step * 0.15)
                cmd_j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, y_sweep, z_sweep], target_orn)
                action_buffer.append((cmd_j, 0.0, target_orn))
                
                exec_j, _, _ = action_buffer[0]
                for i in range(7): 
                    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, exec_j[i], force=200)
                p.stepSimulation()
                
                joints = p.getJointStates(robot_id, range(7))
                traj_qvel.append(np.mean([j[1] for j in joints])) 
            return np.array(traj_qvel)

        # ==============================================================
        # 🎬 场景 2：垂直掂量 Hefting (专攻质量、刚度) - 修复死机与观测度Bug
        # ==============================================================
        elif stage == 2:
            for step in range(150):
                # 改为在空中高频上下抖动，利用 m(g+a) 纯净剥离质量！
                z_heft = 0.15 + 0.05 * np.sin(step * 0.5) 
                cmd_j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, 0.0, z_heft], target_orn)
                action_buffer.append((cmd_j, 20.0, target_orn)) # 死死抓紧
                
                exec_j, exec_force, _ = action_buffer[0]
                for i in range(7): 
                    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, exec_j[i], force=200)
                # 修复夹爪失控Bug：必须执行力控指令
                p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, 0.02, force=exec_force)
                p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, 0.02, force=exec_force)
                p.stepSimulation()
                
                pos, _ = p.getBasePositionAndOrientation(cube_id)
                contacts = p.getContactPoints(robot_id, cube_id)
                traj_z.append(pos[2]) 
                traj_f.append(max([c[9] for c in contacts]) if contacts else 0.0)
            return np.array(traj_z), np.array(traj_f)

        # ==============================================================
        # 🎬 场景 3：激励滑脱与偏心旋转 (专攻摩擦力、质心偏移) - 修复力矩物理Bug
        # ==============================================================
        elif stage == 3:
            # 修复时间断层Bug：逼近阶段的指令也必须压入 Buffer！
            for step in range(50):
                cmd_j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, 0.0, 0.025], target_orn)
                action_buffer.append((cmd_j, 15.0, target_orn))
                
                exec_j, exec_force, _ = action_buffer[0]
                for i in range(7): 
                    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, exec_j[i], force=200)
                p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, 0.02, force=exec_force)
                p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, 0.02, force=exec_force)
                p.stepSimulation()

            for step in range(150):
                z_lift = 0.025 + 0.15 * (step / 150.0)
                roll_angle = np.pi + 0.5 * np.sin(step * 0.1) 
                cmd_orn = p.getQuaternionFromEuler([roll_angle, 0, 0])
                cmd_j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, 0.0, z_lift], cmd_orn)
                
                action_buffer.append((cmd_j, 8.0, cmd_orn)) # 8N 弱抓取引发滑脱
                exec_j, exec_force, exec_orn = action_buffer[0]
                
                for i in range(7): p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, exec_j[i], force=200)
                p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, 0.022, force=exec_force)
                p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, 0.022, force=exec_force)
                
                # 🌟 修复物理力矩Bug：严格利用世界坐标系下的外积计算 3D Torque
                pos, orn = p.getBasePositionAndOrientation(cube_id)
                rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
                # 局部坐标转世界坐标 (自动涵盖了 dx, dy, dz)
                local_com = np.array([params['com_dx'], params['com_dy'], params['com_dz']])
                world_com_offset = rot_mat.dot(local_com) 
                # 真实扭矩 = 力臂(世界) x 重力(世界)
                gravity_force = np.array([0, 0, -9.81 * params['mass']])
                world_torque = np.cross(world_com_offset, gravity_force)
                
                # 在世界坐标系下施加！
                p.applyExternalTorque(cube_id, -1, world_torque, p.WORLD_FRAME)
                
                p.stepSimulation()
                
                euler = p.getEulerFromQuaternion(orn)
                contacts = p.getContactPoints(robot_id, cube_id)
                
                traj_z.append(pos[2])
                traj_roll.append(euler[0]) 
                traj_f.append(max([c[9] for c in contacts]) if contacts else 0.0)
            return np.array(traj_z), np.array(traj_f), np.array(traj_roll)

    finally:
        p.disconnect()

# =====================================================================
# 📈 课程标定 (Curriculum ID) 主管线
# =====================================================================
def main():
    print("\n" + "="*60)
    print("🚀 [Phase 3 工业终极实战] 10维隐性参数课程标定 (物理漏洞修复版)")
    print("="*60)

    # 上帝视角隐藏参数
    TRUE_PARAMS = {
        "mass": 0.15, "mu_lat": 0.80, "mu_spin": 0.05,
        "k_n": 5000.0, "c_n": 50.0,
        "com_dx": 0.01, "com_dy": -0.005, "com_dz": 0.002, # 加入了 com_dz
        "joint_damp": 1.5, "sys_delay": 0.03
    }
    
    NOMINAL_PARAMS = {
        "mass": 0.05, "mu_lat": 0.20, "mu_spin": 0.01,
        "k_n": 1000.0, "c_n": 10.0,
        "com_dx": 0.0, "com_dy": 0.0, "com_dz": 0.0,
        "joint_damp": 0.1, "sys_delay": 0.0
    }

    print("📡 正在采集真实世界 Ground Truth 多模态数据...")
    tgt_1_qvel = simulate_stage(1, TRUE_PARAMS)
    tgt_2_z, tgt_2_f = simulate_stage(2, TRUE_PARAMS)
    tgt_3_z, tgt_3_f, tgt_3_roll = simulate_stage(3, TRUE_PARAMS)

    # 注入噪声
    tgt_1_qvel += np.random.normal(0, 0.02, size=tgt_1_qvel.shape)
    tgt_2_z += np.random.normal(0, 0.001, size=tgt_2_z.shape)
    tgt_2_f += np.random.normal(0, 1.0, size=tgt_2_f.shape)
    tgt_3_z += np.random.normal(0, 0.001, size=tgt_3_z.shape)
    tgt_3_f += np.random.normal(0, 1.0, size=tgt_3_f.shape)
    tgt_3_roll += np.random.normal(0, 0.02, size=tgt_3_roll.shape)

    # 🌟 修复归一化陷阱：加入合理的物理下限阈值防止 NaN
    scale_1_qvel = max(np.var(tgt_1_qvel), 0.001)
    scale_2_z    = max(np.var(tgt_2_z), 0.0001)
    scale_2_f    = max(np.var(tgt_2_f), 1.0)
    scale_3_z    = max(np.var(tgt_3_z), 0.0001)
    scale_3_f    = max(np.var(tgt_3_f), 1.0)
    scale_3_roll = max(np.var(tgt_3_roll), 0.01)

    # 🏫 课程 1：空中乱舞 (阻尼, 延迟)
    print("\n🧠 [Stage 1] 执行空中乱舞，剥离【阻尼与延迟】...")
    def obj_stage1(trial):
        params = NOMINAL_PARAMS.copy()
        params['joint_damp'] = trial.suggest_float("joint_damp", 0.1, 3.0)
        params['sys_delay'] = trial.suggest_float("sys_delay", 0.0, 0.08)
        sim_qvel = simulate_stage(1, params)
        return np.mean((sim_qvel - tgt_1_qvel)**2) / scale_1_qvel

    study1 = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), direction="minimize")
    study1.optimize(obj_stage1, n_trials=40)
    NOMINAL_PARAMS.update(study1.best_params)

    # 🏫 课程 2：垂直掂量 (质量, 刚度, 阻尼)
    print("\n🧠 [Stage 2] 执行垂直掂量，严谨剥离【质量与刚度】...")
    def obj_stage2(trial):
        params = NOMINAL_PARAMS.copy()
        params['mass'] = trial.suggest_float("mass", 0.05, 0.3)
        params['k_n'] = trial.suggest_float("k_n", 1000, 10000)
        params['c_n'] = trial.suggest_float("c_n", 10, 150)
        sim_z, sim_f = simulate_stage(2, params)
        return (np.mean((sim_z - tgt_2_z)**2) / scale_2_z) + 0.5 * (np.mean((sim_f - tgt_2_f)**2) / scale_2_f)

    study2 = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), direction="minimize")
    study2.optimize(obj_stage2, n_trials=50)
    NOMINAL_PARAMS.update(study2.best_params)

    # 🏫 课程 3：扭转滑脱 (摩擦, 质心偏移)
    print("\n🧠 [Stage 3] 执行扭转滑脱，严谨剥离【摩擦系数与全向质心偏移】...")
    def obj_stage3(trial):
        params = NOMINAL_PARAMS.copy()
        params['mu_lat'] = trial.suggest_float("mu_lat", 0.2, 1.2)
        params['mu_spin'] = trial.suggest_float("mu_spin", 0.0, 0.1)
        params['com_dx'] = trial.suggest_float("com_dx", -0.03, 0.03)
        params['com_dy'] = trial.suggest_float("com_dy", -0.03, 0.03)
        params['com_dz'] = trial.suggest_float("com_dz", -0.03, 0.03)
        
        sim_z, sim_f, sim_roll = simulate_stage(3, params)
        loss_z = np.mean((sim_z - tgt_3_z)**2) / scale_3_z
        loss_f = np.mean((sim_f - tgt_3_f)**2) / scale_3_f
        loss_roll = np.mean((sim_roll - tgt_3_roll)**2) / scale_3_roll
        return loss_z + loss_f*0.2 + loss_roll * 1.5

    study3 = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), direction="minimize")
    study3.optimize(obj_stage3, n_trials=60)
    NOMINAL_PARAMS.update(study3.best_params)

    print("\n" + "="*60)
    print("🏆 [Curriculum ID 最终十维反算结果验收]")
    print(f"{'Parameter':<15} | {'Ground Truth':<15} | {'CMA-ES Found':<15} | {'Error %':<10}")
    print("-" * 60)
    for key, true_val in TRUE_PARAMS.items():
        pred_val = NOMINAL_PARAMS[key]
        err_str = "N/A" if abs(true_val) < 1e-5 else f"{abs(pred_val - true_val) / abs(true_val) * 100:.1f}%"
        print(f"{key:<15} | {true_val:<15.4f} | {pred_val:<15.4f} | {err_str:<10}")
    print("="*60)

if __name__ == "__main__":
    main()