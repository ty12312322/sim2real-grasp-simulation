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

    EE_INDEX = 11
    FINGER_L, FINGER_R = 9, 10
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])  # 夹爪垂直向下

    logger = PhysicsLogger()
    sim_time = 0.0

    def step_control(target_pos, finger_pos=0.04, steps=100, record=False):
        nonlocal sim_time
        for _ in range(steps):
            joint_poses = p.calculateInverseKinematics(
                robot_id, EE_INDEX, target_pos, target_orn
            )
            for i in range(7):
                p.setJointMotorControl2(
                    robot_id, i, p.POSITION_CONTROL, targetPosition=joint_poses[i], force=200
                )
            p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, targetPosition=finger_pos, force=20)
            p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, targetPosition=finger_pos, force=20)

            p.stepSimulation()
            sim_time += time_step
            time.sleep(time_step)

            if record:
                contacts = p.getContactPoints(robot_id, cube_id)
                logger.record(sim_time, contacts)

    # 3. 抓取控制状态机
    print(">>> 1. 移动至物体上方...")
    step_control([0.5, 0.0, 0.15], finger_pos=0.04, steps=80)

    print(">>> 2. 下降贴近物体 (调整高度至 0.065m，避免手指砸地)...")
    step_control([0.5, 0.0, 0.065], finger_pos=0.04, steps=80)  # <-- 高度从 0.025 改为 0.065

    print(">>> 3. 闭合夹爪 (开启数据采样)...")
    step_control([0.5, 0.0, 0.065], finger_pos=0.00, steps=120, record=True) # <-- 高度改为 0.065

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