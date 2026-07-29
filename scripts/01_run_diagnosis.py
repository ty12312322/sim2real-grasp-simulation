import os
import time
import numpy as np
import pybullet as p
import pybullet_data


class PhysicsLogger:
    """物理诊断数据采集器：负责高频记录法向力与穿透深度"""

    def __init__(self):
        self.time_stamps = []
        self.normal_forces = []
        self.penetrations = []

    def record(self, t, contacts):
        max_force = 0.0
        max_pen = 0.0

        for c in contacts:
            pen_depth = -c[8]  # Contact Distance (负值即穿透深度，单位: m)
            normal_force = c[9]  # Normal Force (单位: N)

            if normal_force > max_force:
                max_force = normal_force
            if pen_depth > max_pen:
                max_pen = pen_depth

        self.time_stamps.append(t)
        self.normal_forces.append(max_force)
        self.penetrations.append(max_pen * 1000.0)  # 转换为毫米 (mm)

    def print_summary(self, grasp_success):
        forces = np.array(self.normal_forces)
        pens = np.array(self.penetrations)

        print("\n" + "=" * 50)
        print("📊 [Phase 1: 物理仿真缺陷诊断报告]")
        print("=" * 50)
        print(f"1. 抓取最终结果 : {'✅ 成功抬升' if grasp_success else '❌ 抓取失败/滑脱'}")
        print(f"2. 最大法向接触力 : {np.max(forces):.2f} N")
        print(f"3. 接触力波动标准差 : {np.std(forces):.2f} N  <-- [指标: 越高说明高频震荡越严重]")
        print(f"4. 最大几何穿透深度 : {np.max(pens):.4f} mm <-- [指标: 求解器约束硬度]")
        print("=" * 50 + "\n")


def run_simulation():
    # 1. 启动物理引擎
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    time_step = 1.0 / 240.0
    p.setTimeStep(time_step)

    # 2. 加载场景与实体
    p.loadURDF("plane.urdf")
    cube_id = p.loadURDF("cube_small.urdf", basePosition=[0.5, 0.0, 0.025])
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # ------------------ 新增：设置 Franka 初始待命姿态 ------------------
    # 这是 Franka 官方推荐的 Home Pose，带有明显的肘部弯曲
    #PyBullet 里的逆运动学（IK）求解器使用的是数值迭代法（基于雅可比矩阵伪逆），因此初始姿态对求解速度和稳定性有很大影响。
    #全部是0 是singularity，局部最优解不对。
    home_poses = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]
    for i in range(7):
        p.resetJointState(robot_id, i, home_poses[i])
    # ------------------------------------------------------------------

    EE_INDEX = 11  # panda_link8 / hand frame  # 告诉求解器：我们要操纵机械臂的第 11 号节点（夹爪中心点）
    FINGER_L, FINGER_R = 9, 10   # 告诉求解器：这是两根可以滑动的手指关节
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0]) # 把夹爪翻转 180 度，让它垂直指着地面

    # 增加表面滑动摩擦力，防止接触即滑脱
    #p.changeDynamics(cube_id, -1, lateralFriction=1.0)
    #p.changeDynamics(robot_id, FINGER_L, lateralFriction=1.0)
    #p.changeDynamics(robot_id, FINGER_R, lateralFriction=1.0)

    logger = PhysicsLogger()
    sim_time = 0.0

    def step_control(target_pos, finger_pos=0.04, steps=100, record=False):
        #finger_pos=0.04 命令单侧手指往外平移 0.04 m（4 cm）。
        #因为两根手指背道而驰，此时夹爪的总开口宽度就是 8 cm。
        nonlocal sim_time
        for _ in range(steps):
            # 你给出一个三维空间坐标 (target_pos)，求解器帮你反推出 7 个关节分别要转动多少度 (joint_poses)
            joint_poses = p.calculateInverseKinematics(
                robot_id, EE_INDEX, target_pos, target_orn
            )
            # 2. 施加电机扭矩 (驱动 7 个关节)
            #force=Maximum Motor Force  
            # 如果当前位置离目标位置非常远，PD 公式算出来的理想驱动力 $\tau$ 可能会趋于无穷大。
            # 为了防止仿真爆炸（产生无限大的加速度），求解器对输出力矩做了一个硬性截断
            for i in range(7):
                p.setJointMotorControl2(
                    robot_id, i, p.POSITION_CONTROL, targetPosition=joint_poses[i], force=200
                )
            # 3. 驱动夹爪闭合
            p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, targetPosition=finger_pos, force=20)
            p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, targetPosition=finger_pos, force=20)
            # 4. 时间流逝
            p.stepSimulation()
            sim_time += time_step
            time.sleep(time_step)

            if record:
                contacts = p.getContactPoints(robot_id, cube_id)
                logger.record(sim_time, contacts)

    # 3. 抓取控制状态机 # 按照 逼近 -> 下降 -> 闭合 -> 抬升 的逻辑调用核心控制引擎
    print(">>> 1. 移动至物体上方...")
    step_control([0.5, 0.0, 0.15], finger_pos=0.04, steps=80)

    print(">>> 2. 下降贴近物体 ...")
    step_control([0.5, 0.0, 0.025], finger_pos=0.04, steps=80)  # <-- 高度从 0.025 改为 0.065

    print(">>> 3. 闭合夹爪 (开启数据采样)...")
    step_control([0.5, 0.0, 0.025], finger_pos=0.00, steps=120, record=True) # <-- 高度改为 0.065

    print(">>> 4. 尝试抬升 (持续数据采样)...")
    step_control([0.5, 0.0, 0.20], finger_pos=0.00, steps=150, record=True)

    # 4. 评估结果并输出诊断
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    grasp_success = cube_pos[2] > 0.10
    logger.print_summary(grasp_success)

    input("请先在终端按 [Enter] 键关闭仿真窗口...")
    if p.isConnected(): # <-- 增加安全判断，防止手动关窗口报错
        p.disconnect()



if __name__ == "__main__":
    run_simulation()