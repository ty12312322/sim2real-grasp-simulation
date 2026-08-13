import time
import numpy as np
import pybullet as p
import pybullet_data
import optuna
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

def simulate_grasp(mass, friction, gui=False):
    """
    终极黑盒仿真机：采用扫频激励动作，同时返回【位置 Z】、【接触力 F】和【绝对速度 V】
    """
    if gui:
        physics_client = p.connect(p.GUI)
    else:
        physics_client = p.connect(p.DIRECT)
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # 严谨的物理求解器参数
    p.setPhysicsEngineParameter(
        numSolverIterations=100,                              
        numSubSteps=10,                                       
        enableConeFriction=1,
        erp=0.9,             
        globalCFM=1e-5,      
        contactSlop=1e-4     
    )

    p.loadURDF("plane.urdf")
    cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0.0, 0.025])
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # 注入参数
    p.changeDynamics(cube_id, -1, mass=mass, lateralFriction=friction)
    p.changeDynamics(robot_id, 9, lateralFriction=friction)
    p.changeDynamics(robot_id, 10, lateralFriction=friction)

    for i, angle in enumerate([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]):
        p.resetJointState(robot_id, i, angle)

    EE_INDEX = 11  
    FINGER_L, FINGER_R = 9, 10   
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0]) 

    # 1. 逼近并软闭合 (中等力度15N，给滑脱留出微小空间)
    for _ in range(100):
        j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, 0.0, 0.025], target_orn)
        for i in range(7): p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, j[i], force=200)
        p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, 0.02, force=15)
        p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, 0.02, force=15)
        p.stepSimulation()

    # 2. 🌟 核心改动：扫频激励轨迹采集 (Sine Sweep Excitation)
    z_traj, f_traj, v_traj = [], [], []
    for step in range(150):
        # 制造一个随时间变化的波浪形甩动，同时缓慢抬升！
        # 这会产生极其丰富的加速度和速度变化，逼出真实的摩擦力学特征！
        y_sweep = 0.08 * np.sin(step * 0.15) 
        z_lift = 0.025 + 0.15 * (step / 150.0)
        
        j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, y_sweep, z_lift], target_orn)
        for i in range(7): 
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, j[i], force=200)
        p.stepSimulation()
        
        # 采集 1: Z 轴位置
        pos, _ = p.getBasePositionAndOrientation(cube_id)
        z_traj.append(pos[2])
        
        # 采集 2: 夹爪接触法向力
        contacts = p.getContactPoints(robot_id, cube_id)
        max_force = max([c[9] for c in contacts]) if contacts else 0.0
        f_traj.append(max_force)
        
        # 🌟 采集 3: 方块的绝对线速度 (Velocity)
        vel, _ = p.getBaseVelocity(cube_id)
        # 计算三维速度向量的模长 (Magnitude)
        v_magnitude = np.linalg.norm(vel)
        v_traj.append(v_magnitude)

    p.disconnect()
    return np.array(z_traj), np.array(f_traj), np.array(v_traj)


def main():
    print("\n" + "="*55)
    print("🚀 [Phase 2 完全体: 扫频激励 + 动静力学联合标定]")
    print("="*55)

    # 1. 设定 Ground Truth (附加三种传感器的高斯白噪声)
    TRUE_MASS = 0.12
    TRUE_FRICTION = 0.85
    print(f"📡 正在运行正弦扫频轨迹，采集真实世界数据...")
    target_z, target_f, target_v = simulate_grasp(TRUE_MASS, TRUE_FRICTION, gui=False)
    
    target_z += np.random.normal(0, 0.002, size=target_z.shape) # 位置噪声 2mm
    target_f += np.random.normal(0, 0.5, size=target_f.shape)   # 力控底噪 0.5N
    target_v += np.random.normal(0, 0.01, size=target_v.shape)  # 速度估计噪声 0.01m/s

    # ================= 动态计算特征标尺 =================
    scale_z = np.mean(target_z**2) + 1e-6  
    scale_f = np.mean(target_f**2) + 1e-6  
    scale_v = np.mean(target_v**2) + 1e-6  
    print(f"⚖️ 归一化标尺 -> Z轴: {scale_z:.4f} | 受力: {scale_f:.4f} | 速度: {scale_v:.4f}")
    # ====================================================

    # ================= 阶段 1：专攻摩擦力 (Force + Velocity) =================
    print("\n🧠 [Stage 1] 锁定质量假设，CMA-ES 联合【受力+速度】攻坚【摩擦力】...")
    def objective_stage1(trial):
        guess_friction = trial.suggest_float("friction", 0.1, 1.5)
        # 第一阶段，我们用速度波形和受力波形双管齐下！
        _, sim_f, sim_v = simulate_grasp(0.10, guess_friction, gui=False)
        
        loss_f_norm = np.mean((sim_f - target_f)**2) / scale_f
        loss_v_norm = np.mean((sim_v - target_v)**2) / scale_v
        
        return loss_f_norm + loss_v_norm # 动静力学完美融合！

    study_stage1 = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), direction="minimize")
    study_stage1.optimize(objective_stage1, n_trials=30)
    best_friction = study_stage1.best_params['friction']
    print(f"✅ Stage 1 完成！成功剥离摩擦力 -> {best_friction:.4f}")

    # ================= 阶段 2：专攻质量 (Z-Position + Force) =================
    print(f"\n🧠 [Stage 2] 固定摩擦力={best_friction:.4f}，CMA-ES 联合【位置+受力】攻坚【质量】...")
    def objective_stage2(trial):
        guess_mass = trial.suggest_float("mass", 0.01, 0.3)
        sim_z, sim_f, _ = simulate_grasp(guess_mass, best_friction, gui=False)
        
        loss_z_norm = np.mean((sim_z - target_z)**2) / scale_z
        loss_f_norm = np.mean((sim_f - target_f)**2) / scale_f
        
        return loss_z_norm + loss_f_norm

    study_stage2 = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), direction="minimize")
    study_stage2.optimize(objective_stage2, n_trials=30)
    best_mass = study_stage2.best_params['mass']
    print(f"✅ Stage 2 完成！成功剥离质量 -> {best_mass:.4f}")

    # ================= 成绩揭晓 =================
    print("\n" + "="*50)
    print("🏆 [最终完全体标定结果揭晓]")
    print(f"【真实参数】 -> 质量: {TRUE_MASS:.4f} kg, 摩擦: {TRUE_FRICTION:.4f}")
    print(f"【AI反算值】 -> 质量: {best_mass:.4f} kg, 摩擦: {best_friction:.4f}")
    
    mass_err = abs(best_mass - TRUE_MASS) / TRUE_MASS * 100
    fric_err = abs(best_friction - TRUE_FRICTION) / TRUE_FRICTION * 100
    print(f"📉 最终质量误差率: {mass_err:.2f}% | 摩擦力误差率: {fric_err:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()