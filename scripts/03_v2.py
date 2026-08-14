import time
import numpy as np
import pybullet as p
import pybullet_data
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

def simulate_grasp(mass, friction, gui=False):
    """
    改进版仿真机：加入【动态松手】扫频激励，逼出真实的滑动摩擦极限
    """
    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # 严谨的物理求解器参数（增加迭代次数，保证接触稳定）
    p.setPhysicsEngineParameter(
        numSolverIterations=150,
        numSubSteps=10,
        enableConeFriction=1,
        contactSlop=1e-4
    )

    p.loadURDF("plane.urdf")
    cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0.0, 0.025])
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # 注入物理参数
    p.changeDynamics(cube_id, -1, mass=mass, lateralFriction=friction)
    p.changeDynamics(robot_id, 9, lateralFriction=friction)
    p.changeDynamics(robot_id, 10, lateralFriction=friction)

    for i, angle in enumerate([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]):
        p.resetJointState(robot_id, i, angle)

    EE_INDEX = 11  
    FINGER_L, FINGER_R = 9, 10   
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0]) 

    # 1. 逼近并软闭合 (初始给 5N 夹力，不要夹太死，给滑脱创造前提)
    for _ in range(100):
        j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, 0.0, 0.025], target_orn)
        for i in range(7): p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, j[i], force=200)
        p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, 0.02, force=5)
        p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, 0.02, force=5)
        p.stepSimulation()

    # 2. 🌟 核心改动：扫频激励 + 动态力控衰减 (Dynamic Force Release)
    z_traj, f_traj, v_traj = [], [], []
    for step in range(150):
        y_sweep = 0.08 * np.sin(step * 0.15) 
        z_lift = 0.025 + 0.15 * (step / 150.0)
        
        # 🔥 灵魂机制：夹力从 5N 线性衰减到 0.5N
        # 无论物体的摩擦和质量是多少，在这 150 步内，必定会在某一个瞬间发生【滑脱】！
        # 这个滑脱的瞬间，就是摩擦力和质量这两个物理属性的“指纹”。
        dynamic_grip_force = 5.0 - 4.5 * (step / 150.0) 
        
        j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, y_sweep, z_lift], target_orn)
        for i in range(7): 
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, j[i], force=200)
            
        p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, 0.02, force=dynamic_grip_force)
        p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, 0.02, force=dynamic_grip_force)
        p.stepSimulation()
        
        # 采集状态
        pos, _ = p.getBasePositionAndOrientation(cube_id)
        vel, _ = p.getBaseVelocity(cube_id)
        contacts = p.getContactPoints(robot_id, cube_id)
        
        z_traj.append(pos[2])
        v_traj.append(np.linalg.norm(vel)) # 速度在滑脱瞬间会激增
        f_traj.append(max([c[9] for c in contacts]) if contacts else 0.0)

    p.disconnect()
    return np.array(z_traj), np.array(f_traj), np.array(v_traj)


def main():
    print("\n" + "="*60)
    print("🚀 [运行中] 2D参数 CMA-ES 联合寻优 (动态滑脱激励版)")
    print("="*60)

    # 1. 设定 Ground Truth 并采集目标数据
    TRUE_MASS = 0.12
    TRUE_FRICTION = 0.85
    print(f"📡 正在运行物理引擎，采集目标物体特征轨迹...")
    target_z, target_f, target_v = simulate_grasp(TRUE_MASS, TRUE_FRICTION, gui=False)
    
    # 模拟真实世界传感器底噪
    target_z += np.random.normal(0, 0.001, size=target_z.shape) 
    target_f += np.random.normal(0, 0.5, size=target_f.shape)   
    target_v += np.random.normal(0, 0.01, size=target_v.shape)  

    scale_z = np.mean(target_z**2) + 1e-6  
    scale_f = np.mean(target_f**2) + 1e-6  
    scale_v = np.mean(target_v**2) + 1e-6  

    # 2. 🌟 核心改动：废除分步计算，直接进入 2D 联合参数空间
    print("\n🧠 CMA-ES 启动：在 [质量 x 摩擦] 2D空间中寻找全局极小值...")
    
    def objective_unified(trial):
        # 让优化器同时猜两个值，绝不固定任何一个！
        guess_mass = trial.suggest_float("mass", 0.01, 0.3)
        guess_friction = trial.suggest_float("friction", 0.1, 1.5)
        
        sim_z, sim_f, sim_v = simulate_grasp(guess_mass, guess_friction, gui=False)
        
        loss_z = np.mean((sim_z - target_z)**2) / scale_z
        loss_v = np.mean((sim_v - target_v)**2) / scale_v
        loss_f = np.mean((sim_f - target_f)**2) / scale_f
        
        # 🔥 核心改动：由于受力数据(F)尖峰噪音过大，降低其权重
        # 主要依靠 Z轴掉落位置 和 滑脱瞬间的速度激增(V) 来锁定参数
        total_loss = (loss_z * 1.0) + (loss_v * 1.0) + (loss_f * 0.1)
        return total_loss

    # 提高 Trial 数量。参数空间维度上升了，CMA-ES需要足够的种群代数来收敛
    study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), direction="minimize")
    study.optimize(objective_unified, n_trials=150) 
    
    best_mass = study.best_params['mass']
    best_friction = study.best_params['friction']

    # ================= 成绩揭晓 =================
    print("\n" + "="*50)
    print("🏆 [标定结果揭晓]")
    print(f"【真实参数】 -> 质量: {TRUE_MASS:.4f} kg | 摩擦: {TRUE_FRICTION:.4f}")
    print(f"【反算预测】 -> 质量: {best_mass:.4f} kg | 摩擦: {best_friction:.4f}")
    
    mass_err = abs(best_mass - TRUE_MASS) / TRUE_MASS * 100
    fric_err = abs(best_friction - TRUE_FRICTION) / TRUE_FRICTION * 100
    print(f"📉 最终质量误差: {mass_err:.2f}% | 摩擦力误差: {fric_err:.2f}%")
    if mass_err < 5 and fric_err < 5:
        print("✅ 表现完美！参数成功解耦并收敛！")
    print("="*50)

if __name__ == "__main__":
    main()