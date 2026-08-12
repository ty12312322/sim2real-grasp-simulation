import time
import numpy as np
import pybullet as p
import pybullet_data
import optuna
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

def simulate_grasp(mass, friction, gui=False):
    """黑盒仿真机：抓取并返回 Z轴轨迹 与 接触力轨迹"""
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

    # 1. 逼近并闭合 (中等力度，防止过度混沌)
    for _ in range(100):
        j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, 0.0, 0.025], target_orn)
        for i in range(7): p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, j[i], force=200)
        p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, 0.02, force=15)
        p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, 0.02, force=15)
        p.stepSimulation()

    # 2. 抬升采集
    z_traj, f_traj = [], []
    for _ in range(150):
        j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, 0.0, 0.20], target_orn)
        for i in range(7): p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, j[i], force=200)
        p.stepSimulation()
        
        pos, _ = p.getBasePositionAndOrientation(cube_id)
        z_traj.append(pos[2])
        
        contacts = p.getContactPoints(robot_id, cube_id)
        max_force = max([c[9] for c in contacts]) if contacts else 0.0
        f_traj.append(max_force)

    p.disconnect()
    return np.array(z_traj), np.array(f_traj)


def main():
    print("\n" + "="*50)
    print("🚀 [Phase 2: 工业级分步解耦标定 (Curriculum System ID)]")
    print("="*50)

    # 1. 设定 Ground Truth (附带5%的高斯白噪声)
    TRUE_MASS = 0.12
    TRUE_FRICTION = 0.85
    print(f"📡 正在采集真实世界数据...")
    target_z, target_f = simulate_grasp(TRUE_MASS, TRUE_FRICTION, gui=False)
    target_z += np.random.normal(0, 0.002, size=target_z.shape) 
    target_f += np.random.normal(0, 0.5, size=target_f.shape)   

    # ================= 阶段 1：专攻摩擦力 (Force Loss) =================
    print("\n🧠 [Stage 1] 锁定质量假设，CMA-ES 全力攻坚【摩擦力】...")
    def objective_stage1(trial):
        guess_friction = trial.suggest_float("friction", 0.1, 1.5)
        # 假设质量为一个名义值(0.1)，专门对比接触力的分布
        _, sim_f = simulate_grasp(0.10, guess_friction, gui=False)
        return np.mean((sim_f - target_f)**2)

    study_stage1 = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), direction="minimize")
    study_stage1.optimize(objective_stage1, n_trials=30)
    best_friction = study_stage1.best_params['friction']
    print(f"✅ Stage 1 完成！成功剥离摩擦力 -> {best_friction:.4f}")

    # ================= 阶段 2：专攻质量 (Z-Position Loss) =================
    print(f"\n🧠 [Stage 2] 固定摩擦力={best_friction:.4f}，CMA-ES 全力攻坚【质量】...")
    def objective_stage2(trial):
        guess_mass = trial.suggest_float("mass", 0.01, 0.3)
        # 使用第一阶段标定好的真实摩擦力，现在唯一未知的就是质量！
        sim_z, _ = simulate_grasp(guess_mass, best_friction, gui=False)
        return np.mean((sim_z - target_z)**2)

    study_stage2 = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), direction="minimize")
    study_stage2.optimize(objective_stage2, n_trials=30)
    best_mass = study_stage2.best_params['mass']
    print(f"✅ Stage 2 完成！成功剥离质量 -> {best_mass:.4f}")

    # ================= 成绩揭晓 =================
    print("\n" + "="*50)
    print("🏆 [最终课程标定结果揭晓]")
    print(f"【真实参数】 -> 质量: {TRUE_MASS:.4f} kg, 摩擦: {TRUE_FRICTION:.4f}")
    print(f"【AI反算值】 -> 质量: {best_mass:.4f} kg, 摩擦: {best_friction:.4f}")
    
    mass_err = abs(best_mass - TRUE_MASS) / TRUE_MASS * 100
    fric_err = abs(best_friction - TRUE_FRICTION) / TRUE_FRICTION * 100
    print(f"📉 质量误差率: {mass_err:.2f}% | 摩擦力误差率: {fric_err:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()