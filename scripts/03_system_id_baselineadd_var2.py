import time
import numpy as np
import pybullet as p
import pybullet_data
import optuna
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

def simulate_grasp(mass, friction, gui=False):
    """
    多模态黑盒仿真：引入边缘滑脱动作，同时返回【高度轨迹】和【接触力轨迹】
    """
    if gui:
        physics_client = p.connect(p.GUI)
    else:
        physics_client = p.connect(p.DIRECT)
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

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

    # 1. 逼近物体
    for _ in range(100):
        j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, 0.0, 0.025], target_orn)
        for i in range(7): p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, j[i], force=200)
        p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, 0.04, force=20)
        p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, 0.04, force=20)
        p.stepSimulation()

    # 2. 软弱闭合 (制造滑脱边缘：力量从 20N 降到 5N，间距不完全闭死)
    for _ in range(100):
        p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, 0.022, force=5)
        p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, 0.022, force=5)
        p.stepSimulation()

    # 3. 激励轨迹采集 (暴力抬升 + 侧向甩动)
    z_traj, f_traj = [], []
    for step in range(150):
        # 制造一个 Y 轴的横向甩动扰动
        y_shake = 0.05 * np.sin(step * 0.1) 
        j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, y_shake, 0.20], target_orn)
        for i in range(7): 
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, j[i], force=200)
        
        p.stepSimulation()
        
        # 采集 Z 轴位置
        pos, _ = p.getBasePositionAndOrientation(cube_id)
        z_traj.append(pos[2])
        
        # 采集夹爪法向力
        contacts = p.getContactPoints(robot_id, cube_id)
        max_force = max([c[9] for c in contacts]) if contacts else 0.0
        f_traj.append(max_force)

    p.disconnect()
    return np.array(z_traj), np.array(f_traj)


def main():
    print("\n" + "="*50)
    print("🚀 [Phase 2: 终极解耦系统辨识 (多模态+激励滑脱+自适应归一化)]")
    print("="*50)

    TRUE_MASS = 0.12
    TRUE_FRICTION = 0.85
    
    print(f"正在生成真实世界轨迹 (目标质量: {TRUE_MASS}kg, 目标摩擦: {TRUE_FRICTION})...")
    target_z, target_f = simulate_grasp(TRUE_MASS, TRUE_FRICTION, gui=False)
    
    # 加入真实世界的传感器噪声
    target_z += np.random.normal(0, 0.002, size=target_z.shape) # 相机噪声 2mm
    target_f += np.random.normal(0, 0.5, size=target_f.shape)   # 力传感器底噪 0.5N

    # ================= 🌟核心改动 1：动态计算特征标尺 =================
    # 防止分母为 0，加一个极小的常数 1e-6
    scale_z = np.mean(target_z**2) + 1e-6  
    scale_f = np.mean(target_f**2) + 1e-6  
    print(f"⚖️ 自动计算归一化标尺 -> Z轴方差: {scale_z:.4f}, 受力方差: {scale_f:.2f}")
    # ===================================================================

    history_loss = []
    
    def objective(trial):
        guess_mass = trial.suggest_float("mass", 0.01, 0.3)
        guess_friction = trial.suggest_float("friction", 0.1, 1.5)
        
        sim_z, sim_f = simulate_grasp(guess_mass, guess_friction, gui=False)
        
        # ================= 🌟核心改动 2：无量纲化 Loss =================
        # 将均方误差除以它们各自的标尺，转化为纯粹的“相对畸变率”
        loss_z_norm = np.mean((sim_z - target_z)**2) / scale_z
        loss_f_norm = np.mean((sim_f - target_f)**2) / scale_f
        
        # 现在它们都在 0~1 的量级，直接 1:1 相加即可完美平衡！
        total_loss = loss_z_norm + loss_f_norm
        # ==========================================================
        
        history_loss.append(total_loss)
        return total_loss

    print("\n🧠 CMA-ES 正在撕裂物理耦合，进行 50 代进化...")
    start_time = time.time()
    
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective, n_trials=50)
    
    print(f"\n✅ 优化完成！总耗时: {time.time() - start_time:.2f} 秒")
    print("="*50)
    print("🎯 [终极 CMA-ES 标定结果]")
    print(f"【真实参数】 -> 质量: {TRUE_MASS:.4f} kg, 摩擦: {TRUE_FRICTION:.4f}")
    print(f"【AI反算值】 -> 质量: {study.best_params['mass']:.4f} kg, 摩擦: {study.best_params['friction']:.4f}")
    
    # 计算相对误差率
    mass_err = abs(study.best_params['mass'] - TRUE_MASS) / TRUE_MASS * 100
    fric_err = abs(study.best_params['friction'] - TRUE_FRICTION) / TRUE_FRICTION * 100
    print(f"📉 质量误差率: {mass_err:.2f}% | 摩擦力误差率: {fric_err:.2f}%")
    print("="*50)

    plt.figure(figsize=(8, 5))
    plt.plot(history_loss, color='red', linewidth=2)
    plt.yscale('log') 
    plt.title("Normalized Multi-modal System ID Convergence")
    plt.xlabel("Trial (Iteration)")
    plt.ylabel("Normalized MSE Loss (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()