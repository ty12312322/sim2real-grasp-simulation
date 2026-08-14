import time
import numpy as np
import pybullet as p
import pybullet_data
import optuna
import matplotlib.pyplot as plt
from collections import deque

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 🛠️ 核心引擎：支持 10 维参数注入的终极黑盒仿真器
# =====================================================================
def simulate_grasp_10d(params, gui=False):
    """
    注入 10 维物理/系统参数，返回多模态时间序列数据
    """
    # 1. 解包 10 个隐藏参数 (白盒信任，黑盒辨识)
    (mass, mu_lat, mu_spin, k_n, c_n, 
     com_dx, com_dy, com_dz, 
     joint_damp, sys_delay) = params

    physics_client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    # 提高求解器精度，应对高刚度(k_n)带来的数值爆炸
    p.setPhysicsEngineParameter(numSolverIterations=150, numSubSteps=10)

    p.loadURDF("plane.urdf")
    cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0.0, 0.025])
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    EE_INDEX = 11  
    FINGER_L, FINGER_R = 9, 10   

    # ================= 注入物理参数 =================
    # 1. 物体属性 (质量, 摩擦, 接触刚度, 接触阻尼, 扭转摩擦)
    p.changeDynamics(cube_id, -1, 
                     mass=mass, 
                     lateralFriction=mu_lat,
                     spinningFriction=mu_spin, # 扭转摩擦 (防止物体在夹爪里转动)
                     contactStiffness=k_n,     # 法向接触刚度
                     contactDamping=c_n)       # 法向接触阻尼

    # 2. 机械臂系统属性 (手指摩擦，关节黏性阻尼)
    p.changeDynamics(robot_id, FINGER_L, lateralFriction=mu_lat)
    p.changeDynamics(robot_id, FINGER_R, lateralFriction=mu_lat)
    for i in range(7):
        # 模拟 Stribeck 摩擦中的 Viscous Damping
        p.changeDynamics(robot_id, i, jointDamping=joint_damp)
        p.resetJointState(robot_id, i, 0.0)

    # ================= 初始化系统延迟 Buffer =================
    # 真实系统中的通信延迟：代码下发指令，电机几个控制周期后才执行
    delay_steps = int(sys_delay * 240) # 将延迟时间(s)转化为物理帧数
    action_buffer = deque(maxlen=max(1, delay_steps))
    
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0]) 
    initial_j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, 0.0, 0.025], target_orn)
    for _ in range(max(1, delay_steps)): 
        action_buffer.append((initial_j, 5.0)) # 填充初始指令

    # 数据采集容器
    traj_z, traj_f, traj_qvel, traj_roll = [], [], [], []

    # ================= 执行扫频与抓取测试 =================
    for step in range(200):
        # 1. 算法层生成目标指令 (动态力控 + 扫频运动)
        y_sweep = 0.08 * np.sin(step * 0.15) 
        z_lift = 0.025 + 0.15 * (step / 200.0)
        cmd_force = 5.0 - 4.8 * (step / 200.0) # 逐渐松手逼出摩擦极限
        cmd_j = p.calculateInverseKinematics(robot_id, EE_INDEX, [0.5, y_sweep, z_lift], target_orn)
        
        # 将当前指令压入延迟队列
        action_buffer.append((cmd_j, cmd_force))
        
        # 2. 底层电机执行滞后指令 (模拟 System Delay)
        exec_j, exec_force = action_buffer[0]
        
        for i in range(7): 
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, exec_j[i], force=200)
        p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, 0.02, force=exec_force)
        p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, 0.02, force=exec_force)
        
        # 3. 模拟质心偏移 (CoM Offset) 的物理效应
        # 在 PyBullet 中不重建 URDF 模拟质心偏移的高阶技巧：施加与重力成比例的扭矩
        # Tau = r x F_g
        torque_x = com_dy * (-9.81 * mass)
        torque_y = -com_dx * (-9.81 * mass)
        p.applyExternalTorque(cube_id, -1, [torque_x, torque_y, 0], p.LINK_FRAME)

        p.stepSimulation()
        
        # 4. 采集多模态数据
        pos, orn = p.getBasePositionAndOrientation(cube_id)
        euler = p.getEulerFromQuaternion(orn)
        contacts = p.getContactPoints(robot_id, cube_id)
        joints = p.getJointStates(robot_id, range(7))
        
        traj_z.append(pos[2])                     # 位置
        traj_roll.append(euler[0])                # 翻滚角 (捕捉偏心扭转滑脱)
        traj_f.append(max([c[9] for c in contacts]) if contacts else 0.0) # 接触力
        traj_qvel.append(np.mean([j[1] for j in joints])) # 关节平均角速度 (捕捉阻尼特性)

    p.disconnect()
    return np.array(traj_z), np.array(traj_f), np.array(traj_qvel), np.array(traj_roll)

# =====================================================================
# 📈 CMA-ES 联合优化与可视化主程序
# =====================================================================
def main():
    print("🚀 [Phase 3 工业级实战] 10维隐性参数 CMA-ES 联合标定启动！")
    
    # 1. 定义 Ground Truth (真实系统隐藏参数)
    TRUE_PARAMS = [
        0.15,   # 1. 质量 Mass (kg)
        0.80,   # 2. 切向摩擦 Mu
        0.05,   # 3. 扭转摩擦 Spin Mu (极其难标)
        5000.0, # 4. 接触刚度 Contact Stiffness
        50.0,   # 5. 接触阻尼 Contact Damping
        0.01,   # 6. 质心偏移 X (m) -> 导致左倾
        -0.005, # 7. 质心偏移 Y (m) -> 导致前倾
        0.0,    # 8. 质心偏移 Z (m)
        1.5,    # 9. 关节黏性阻尼 Joint Damping
        0.03    # 10. 系统延迟 System Delay (秒, 约 7 帧)
    ]
    
    print("📡 正在采集真实世界 Ground Truth 多模态数据...")
    tgt_z, tgt_f, tgt_qvel, tgt_roll = simulate_grasp_10d(TRUE_PARAMS, gui=False)
    
    # 注入真实传感器高斯噪声
    tgt_z += np.random.normal(0, 0.001, size=tgt_z.shape) 
    tgt_f += np.random.normal(0, 1.0, size=tgt_f.shape)   
    tgt_roll += np.random.normal(0, 0.02, size=tgt_roll.shape)
    tgt_qvel += np.random.normal(0, 0.05, size=tgt_qvel.shape)

    # 2. ⚖️ 核心：自适应动态归一化 (Dynamic Normalization)
    # 否则 5000刚度的 Loss 会把 0.01偏移的 Loss 完全吞没！
    scale_z = np.var(tgt_z) + 1e-6
    scale_f = np.var(tgt_f) + 1e-6
    scale_roll = np.var(tgt_roll) + 1e-6
    scale_qvel = np.var(tgt_qvel) + 1e-6

    loss_history = []

    def objective(trial):
        # 10 维连续搜索空间定义
        p_mass = trial.suggest_float("mass", 0.05, 0.3)
        p_mu = trial.suggest_float("mu_lat", 0.1, 1.5)
        p_spin = trial.suggest_float("mu_spin", 0.0, 0.1)
        p_kn = trial.suggest_float("k_n", 1000, 10000)
        p_cn = trial.suggest_float("c_n", 10, 200)
        p_dx = trial.suggest_float("com_dx", -0.05, 0.05)
        p_dy = trial.suggest_float("com_dy", -0.05, 0.05)
        p_dz = trial.suggest_float("com_dz", -0.05, 0.05)
        p_damp = trial.suggest_float("joint_damp", 0.1, 5.0)
        p_delay = trial.suggest_float("sys_delay", 0.0, 0.1)

        guess = [p_mass, p_mu, p_spin, p_kn, p_cn, p_dx, p_dy, p_dz, p_damp, p_delay]
        
        # 运行仿真
        sim_z, sim_f, sim_qvel, sim_roll = simulate_grasp_10d(guess, gui=False)
        
        # 归一化 MSE 损失计算
        loss_z = np.mean((sim_z - tgt_z)**2) / scale_z
        loss_f = np.mean((sim_f - tgt_f)**2) / scale_f
        loss_roll = np.mean((sim_roll - tgt_roll)**2) / scale_roll
        loss_qvel = np.mean((sim_qvel - tgt_qvel)**2) / scale_qvel
        
        # 领域知识权重分配：
        # 运动学(位置/旋转)置信度高，动力学(单点力)噪声大置信度低
        total_loss = (loss_z * 1.0) + (loss_roll * 1.5) + (loss_qvel * 0.8) + (loss_f * 0.2)
        
        loss_history.append(total_loss)
        return total_loss

    print("\n🧠 启动 CMA-ES，在 10 维空间寻找超曲面极小值...")
    # 工业界对于 10 维参数，CMA-ES 至少需要 500-1000 次 trial 才能完全收敛。
    # 这里为了让你快速看到效果，设为 200 次（运行约 1-2 分钟）。
    study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), direction="minimize")
    study.optimize(objective, n_trials=200) 

    # 3. 📊 绘制 Loss 收敛图 (你的要求)
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='#1f77b4', alpha=0.8, linewidth=2)
    plt.yscale('log') # 这种系统辨识必定要用对数坐标系看收敛
    plt.title('CMA-ES 10-Dimensional System ID Convergence')
    plt.xlabel('Trial Generation')
    plt.ylabel('Normalized Composite Loss (Log Scale)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('CMA-ES_Loss_Convergence.png', dpi=300)
    print("\n📈 Loss 收敛曲线已保存为: CMA-ES_Loss_Convergence.png")

    # 4. 成绩打印
    print("\n" + "="*60)
    print("🏆 [10维参数反算结果对比]")
    print(f"{'Parameter':<20} | {'Ground Truth':<15} | {'CMA-ES Found':<15}")
    print("-" * 60)
    
    keys = ["mass", "mu_lat", "mu_spin", "k_n", "c_n", "com_dx", "com_dy", "com_dz", "joint_damp", "sys_delay"]
    for i, key in enumerate(keys):
        true_val = TRUE_PARAMS[i]
        pred_val = study.best_params[key]
        print(f"{key:<20} | {true_val:<15.4f} | {pred_val:<15.4f}")
    print("="*60)

if __name__ == "__main__":
    main()