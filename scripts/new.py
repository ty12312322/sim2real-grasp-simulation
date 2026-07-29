import os
import time
import numpy as np
import pybullet as p
import pybullet_data


class PhysicsLogger:
    def __init__(self):
        self.normal_forces = []
        self.penetrations = []

    def record(self, contacts):
        max_force = 0.0
        max_pen = 0.0
        for c in contacts:
            pen_depth = -c[8]
            normal_force = c[9]
            if normal_force > max_force:
                max_force = normal_force
            if pen_depth > max_pen:
                max_pen = pen_depth
        self.normal_forces.append(max_force)
        self.penetrations.append(max_pen * 1000.0)

    def print_summary(self, grasp_success):
        forces = np.array(self.normal_forces) if self.normal_forces else [0.0]
        pens = np.array(self.penetrations) if self.penetrations else [0.0]

        print("\n" + "=" * 50)
        print("📊 [Phase 1: 物理仿真缺陷诊断报告]")
        print("=" * 50)
        print(f"1. 抓取最终结果 : {'✅ 成功抬升' if grasp_success else '❌ 抓取失败/滑脱'}")
        print(f"2. 最大法向接触力 : {np.max(forces):.2f} N")
        print(f"3. 接触力波动标准差 : {np.std(forces):.2f} N")
        print(f"4. 最大几何穿透深度 : {np.max(pens):.4f} mm")
        print("=" * 50 + "\n")


def run_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    time_step = 1.0 / 240.0
    p.setTimeStep(time_step)

    # 视距拉近对准方块
    p.resetDebugVisualizerCamera(
        cameraDistance=0.4, cameraYaw=45, cameraPitch=-15, cameraTargetPosition=[0.5, 0.0, 0.05]
    )

    p.loadURDF("plane.urdf")
    
    # 方块位置：(0.5, 0.0, 0.025)，中心高度 2.5cm
    cube_pos = [0.5, 0.0, 0.025]
    cube_id = p.loadURDF("cube_small.urdf", basePosition=cube_pos)
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    EE_INDEX = 11  # panda_link8 / hand frame  # 告诉求解器：我们要操纵机械臂的第 11 号节点（夹爪中心点）
    FINGER_L, FINGER_R = 9, 10   # 告诉求解器：这是两根可以滑动的手指关节
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0]) # 把夹爪翻转 180 度，让它垂直指着地面

    # 摩擦力与阻尼调整，防止刚接触就滑脱
    p.changeDynamics(cube_id, -1, lateralFriction=1.2, spinningFriction=0.1)
    p.changeDynamics(robot_id, FINGER_L, lateralFriction=1.2)
    p.changeDynamics(robot_id, FINGER_R, lateralFriction=1.2)

    # 关键：手腕到指尖中心的 Z 轴补偿量 (约 10.3cm)
    FINGERTIP_OFFSET = 0.1034 

    logger = PhysicsLogger()

    def move_to_target(ee_target_pos, finger_pos=0.04, steps=100, record=False):
        # 绘制红色的视觉对准目标点，方便在 GUI 中查看对准情况
        p.addUserDebugLine(
            ee_target_pos, [ee_target_pos[0], ee_target_pos[1], 0], [1, 0, 0], lifeTime=0.5, lineWidth=2
        )
        
        for _ in range(steps):
            joint_poses = p.calculateInverseKinematics(
                robot_id, EE_INDEX, ee_target_pos, target_orn,
                maxNumIterations=100, residualThreshold=1e-5
            )
            for i in range(7):
                p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=joint_poses[i], force=250)
            p.setJointMotorControl2(robot_id, FINGER_L, p.POSITION_CONTROL, targetPosition=finger_pos, force=50)
            p.setJointMotorControl2(robot_id, FINGER_R, p.POSITION_CONTROL, targetPosition=finger_pos, force=50)

            p.stepSimulation()
            time.sleep(time_step)

            if record:
                contacts = p.getContactPoints(robot_id, cube_id)
                logger.record(contacts)

    # 计算精准的末端执行器（Hand Frame）目标位置
    grasp_z = cube_pos[2] + FINGERTIP_OFFSET  # 指尖对准方块中心时，手腕应在的高度 (约 0.1284m)
    approach_z = grasp_z + 0.10                # 预抓取高度 (约 0.2284m)
    lift_z = grasp_z + 0.20                    # 抬升高度

    print(">>> [Step 1] 移动至方块正上方...")
    move_to_target([0.5, 0.0, approach_z], finger_pos=0.04, steps=80)

    print(">>> [Step 2] 下降，让手指恰好包覆方块中心...")
    move_to_target([0.5, 0.0, grasp_z], finger_pos=0.04, steps=80)

    print(">>> [Step 3] 闭合夹爪 (开始监控接触物理数据)...")
    move_to_target([0.5, 0.0, grasp_z], finger_pos=0.00, steps=100, record=True)

    print(">>> [Step 4] 向上抬升...")
    move_to_target([0.5, 0.0, lift_z], finger_pos=0.00, steps=120, record=True)

    # 4. 结果评估
    final_cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    grasp_success = final_cube_pos[2] > 0.10
    logger.print_summary(grasp_success)

    input("请在终端按 [Enter] 键关闭仿真窗口...")
    if p.isConnected():
        p.disconnect()


if __name__ == "__main__":
    run_simulation()