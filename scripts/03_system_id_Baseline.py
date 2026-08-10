import time
import numpy as np
import pybullet as p
import pybullet_data
import optuna
import matplotlib.pyplot as plt

# 关闭 Optuna 的长串终端输出，保持清爽
optuna.logging.set_verbosity(optuna.logging.WARNING)

def simulate_grasp(mass, friction, gui=False):
    """
    黑盒仿真函数：传入质量和摩擦力，输出抓取过程中的高度轨迹。
    gui=False 时为极速无头模式，用于 AI 疯狂试错。
    """
    if gui:
        physics_client = p.connect(p.GUI)
    else:
        physics_client = p.connect(p.DIRECT)  # 极速后台模式，提速百倍！
        
    p.setAdditionalSearchPath(pybullet_data.getDataPath())#PyBullet 官方提供的一大堆测试模型
    p.setGravity(0, 0, -9.81)
    time_step = 1.0 / 240.0
    p.setTimeStep(time_step)

    # 沿用之前的稳定物理参数
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

    # ================= 核心：修改物理参数 =================
    # 改变方块质量
    p.changeDynamics(cube_id, -1, mass=mass)
    # 改变方块与夹爪的摩擦系数
    p.changeDynamics(cube_id, -1, lateralFriction=friction)
    p.changeDynamics(robot_id, 9, lateralFriction=friction)
    p.changeDynamics(robot_id, 10, lateralFriction=friction)
    # ======================================================

    home_poses = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]
    for i in range(7):
        p.resetJointState(robot_id, i, home_poses[i])

    EE_INDEX = 11  
    FINGER_L, FINGER_R = 9, 10   
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0]) 

    def step_control(target_pos, finger_pos, steps):
        for _ in range(steps):
            joint_poses = p.calculateInverseKinematics(robot_id, EE_INDEX, target_pos, target_orn)
            for i in range(7):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_poses[i], force=200)
            p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, targetPosition=finger_pos, force=20)
            p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, targetPosition=finger_pos, force=20)
            p.stepSimulation()

    # 1. 移动、下降、闭合
    step_control([0.5, 0.0, 0.15], finger_pos=0.04, steps=80)
    step_control([0.5, 0.0, 0.025], finger_pos=0.04, steps=80)
    step_control([0.5, 0.0, 0.025], finger_pos=0.02, steps=120)

    # 2. 尝试抬升 (核心数据采集阶段)
    z_trajectory = []
    for _ in range(150):
        # 保持抬升动作
        joint_poses = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, 0.0, 0.20], target_orn)
        for i in range(7):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_poses[i], force=200)
        p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, targetPosition=0.02, force=20)
        p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, targetPosition=0.02, force=20)
        p.stepSimulation()
        
        # 记录每一帧方块的Z轴高度
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        z_trajectory.append(cube_pos[2])

    p.disconnect()
    return np.array(z_trajectory)


def main():
    print("\n" + "="*50)
    print("🚀 [Phase 2: CMA-ES 物理参数系统辨识启动]")
    print("="*50)

    # 1. 设定“真实世界”的秘密参数 (Ground Truth)
    # 我们假装现实世界里，方块质量是 0.12 kg，摩擦系数是 0.85
    TRUE_MASS = 0.12
    TRUE_FRICTION = 0.85
    
    print(f"正在生成真实世界轨迹 (目标质量: {TRUE_MASS}kg, 目标摩擦: {TRUE_FRICTION})...")
    # 生成目标轨迹（加一点极其微小的高斯噪声，模拟现实传感器的误差）
    target_trajectory = simulate_grasp(TRUE_MASS, TRUE_FRICTION, gui=False)
    target_trajectory += np.random.normal(0, 0.001, size=target_trajectory.shape)

    # 用于记录优化过程的数据
    history_loss = []
    
    # 2. 定义目标函数 (Loss Function)
    def objective(trial):
        # 让 CMA-ES 算法在给定的物理范围内“盲猜”参数
        guess_mass = trial.suggest_float("mass", 0.01, 0.3)
        guess_friction = trial.suggest_float("friction", 0.1, 1.5)
        
        # 用猜的参数跑一次极速仿真
        sim_trajectory = simulate_grasp(guess_mass, guess_friction, gui=False)
        
        # 计算 Loss (均方误差 MSE)
        loss = np.mean((sim_trajectory - target_trajectory)**2)
        
        history_loss.append(loss)
        return loss

    # 3. 启动 CMA-ES 优化引擎
    print("\n🧠 CMA-ES 正在黑盒空间中进行 50 代物理进化...")
    start_time = time.time()
    
    # 挂载 CMA-ES 采样器
    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(sampler=sampler, direction="minimize")
    
    # 疯狂试错 50 次
    study.optimize(objective, n_trials=50)
    
    print(f"\n✅ 优化完成！总耗时: {time.time() - start_time:.2f} 秒")
    print("="*50)
    print("🎯 [CMA-ES 标定结果揭晓]")
    print(f"真实参数 -> 质量: {TRUE_MASS:.4f} kg, 摩擦: {TRUE_FRICTION:.4f}")
    print(f"AI反算结果 -> 质量: {study.best_params['mass']:.4f} kg, 摩擦: {study.best_params['friction']:.4f}")
    print(f"最终轨迹均方误差(Loss): {study.best_value:.6f}")
    print("="*50)

    # 4. 画出 Loss 下降曲线
    plt.figure(figsize=(8, 5))
    plt.plot(history_loss, color='blue', linewidth=2)
    plt.yscale('log') # 因为Loss下降极快，用对数坐标轴看得更清楚
    plt.title("CMA-ES System Identification Convergence (Loss)")
    plt.xlabel("Trial (Iteration)")
    plt.ylabel("MSE Loss (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    