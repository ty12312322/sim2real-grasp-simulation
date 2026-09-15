"""
探针实验 06：验证 mu_spin (spinningFriction) 的可观测性与激发方式
==================================================================
背景：
  - 全量标定中 mu_spin 误差 128.66%，估计值触上边界 0.12，完全不可辨识。
  - 两个假设：
    (H1) PyBullet 单刚体求解器 bug：spinningFriction 只在 rollingFriction > 0 时生效
         (bullet3 issue #4188)。当前 05_final.set_params 未设 rollingFriction，
         若接触走单刚体路径，则 mu_spin 注入是空操作。
    (H2) Stage C 的整体姿态摆动让立方体跟随末端一起转，接触面没有绕法线的
         相对扭转，扭转摩擦无法产生可观测阻力矩。

实验 1（平面扭转衰减）：验证 H1
  - 立方体平放平面，绕 z 轴（接触法线）给初始角速度，测角速度衰减。
  - 4 组 (spin, roll)：若 spin=0.1/roll=0 与 spin=0/roll=0 衰减一致、
    仅 spin=0.1/roll=0.1 才更快 → 命中 H1。

实验 2（夹持扭转力矩）：验证 H2
  - Panda 抓取立方体，末端绕接触法线(x轴)小幅扭转，测腕力矩。
  - 对比 mu_spin ∈ {0.005, 0.05, 0.12}：若腕力矩几乎无差异 → 命中 H2。
"""

import os
import importlib.util

import numpy as np
import pybullet as p
import pybullet_data

# 复用 05_final 的 RobotSimulator（实验 2 用），避免复制抓取/IK/接触逻辑
_FINAL05_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_final.py")
_spec = importlib.util.spec_from_file_location("final05", _FINAL05_PATH)
_final05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_final05)
RobotSimulator = _final05.RobotSimulator


def _new_world():
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    p.setPhysicsEngineParameter(
        numSolverIterations=150, numSubSteps=4, enableConeFriction=1,
        contactBreakingThreshold=0.001,
    )
    plane_id = p.loadURDF("plane.urdf")
    # 立方体半高 0.025，底面刚好贴平面
    cube_id = p.loadURDF("cube_small.urdf", basePosition=[0, 0, 0.025])
    return client, plane_id, cube_id


def _set_cube_dynamics(cube_id, spin, roll):
    p.changeDynamics(
        cube_id, -1,
        lateralFriction=0.8,
        spinningFriction=spin,
        rollingFriction=roll,
    )


def experiment_1_torsional_decay():
    """实验 1：平面扭转衰减，诊断 spinningFriction 生效性 / rollingFriction bug。"""
    print("\n" + "=" * 72)
    print("实验 1：平面扭转衰减（诊断 spinningFriction 独立生效性 / rollingFriction bug）")
    print("=" * 72)
    omega0 = 10.0
    steps = 200

    for spin, roll in [(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1)]:
        client, plane_id, cube_id = _new_world()
        _set_cube_dynamics(cube_id, spin, roll)
        for _ in range(80):  # settle：让立方体稳定接触平面
            p.stepSimulation()
        p.resetBaseVelocity(cube_id, [0, 0, 0], [0, 0, omega0])
        omegas = []
        for _ in range(steps):
            p.stepSimulation()
            _lin, ang = p.getBaseVelocity(cube_id)
            omegas.append(ang[2])
        p.disconnect(client)
        omegas = np.array(omegas)
        half_step = None
        for i, w in enumerate(omegas):
            if abs(w) <= 0.5 * omega0:
                half_step = i
                break
        print(f"  spin={spin:.1f} roll={roll:.1f}: "
              f"ω0={omegas[0]:+.3f}  ω_end={omegas[-1]:+.3f}  "
              f"|ω_end/ω0|={abs(omegas[-1])/omega0:.4f}  "
              f"半衰步数={half_step if half_step is not None else '>200'}")


def experiment_2_grasp_torsion():
    """实验 2：夹持扭转力矩，诊断 Stage C 整体旋转是否产生 mu_spin 信号。"""
    print("\n" + "=" * 72)
    print("实验 2：夹持扭转力矩（诊断整体旋转是否产生绕法线的扭转信号）")
    print("=" * 72)

    sim = RobotSimulator(gui=False)
    sim.connect()

    base_params = {
        "mass": 0.15, "mu_lat": 0.80, "mu_spin": 0.05,
        "k_n": 5000.0, "c_n": 50.0,
        "com_dx": 0.005, "com_dy": -0.005, "com_dz": 0.002,
        "joint_damp": 1.5, "sys_delay": 0.025,
    }

    for mu_spin in [0.005, 0.05, 0.12]:
        params = dict(base_params, mu_spin=mu_spin)
        sim.set_params(params)
        sim._settle_grasp(hold_pos=(0.5, 0.0, 0.2), finger_target=0.020, finger_force=3.5, steps=50)

        # 确认抓取成功（立方体应仍被夹在空中，未掉落）
        pos, _ = p.getBasePositionAndOrientation(sim.cube_id)

        steps = int(2.0 * 240)
        times = np.linspace(0, 2.0, steps)
        roll = np.pi + 0.3 * np.sin(2 * np.pi * 0.5 * times)  # 绕 x 轴(接触法线)扭转
        wrist_tau = []
        for i in range(steps):
            orn = p.getQuaternionFromEuler([roll[i], 0.0, 0.0])
            cmd_j = p.calculateInverseKinematics(
                sim.robot_id, sim.EE_INDEX, [0.5, 0.0, 0.20], orn, maxNumIterations=100)
            sim._apply_action(cmd_j, finger_pos=0.020, finger_force=3.5, arm_force=150.0)
            p.stepSimulation()
            states = p.getJointStates(sim.robot_id, [5, 6])
            wrist_tau.append([states[0][3], states[1][3]])
        wrist_tau = np.array(wrist_tau)
        print(f"  mu_spin={mu_spin:.3f}: 抓取后立方体 z={pos[2]:.4f} | "
              f"腕力矩 tau5 std={wrist_tau[:, 0].std():.4f} "
              f"tau6 std={wrist_tau[:, 1].std():.4f} | "
              f"|tau5|均值={np.abs(wrist_tau[:, 0]).mean():.4f}")

    sim.disconnect()


def experiment_3_decay_curve_sensitivity():
    """实验 3：mu_spin 扫描，测完整角速度衰减曲线，评估标定信号的区分度。"""
    print("\n" + "=" * 72)
    print("实验 3：mu_spin 扫描衰减曲线（评估信号在 (0.005, 0.12) 内的区分度）")
    print("=" * 72)
    omega0 = 10.0
    steps = 100
    curves = {}
    for mu_spin in [0.005, 0.02, 0.05, 0.08, 0.12]:
        client, plane_id, cube_id = _new_world()
        p.changeDynamics(cube_id, -1, lateralFriction=0.8, spinningFriction=mu_spin, rollingFriction=0.0)
        for _ in range(80):
            p.stepSimulation()
        p.resetBaseVelocity(cube_id, [0, 0, 0], [0, 0, omega0])
        omegas = []
        for _ in range(steps):
            p.stepSimulation()
            _lin, ang = p.getBaseVelocity(cube_id)
            omegas.append(ang[2])
        p.disconnect(client)
        curves[mu_spin] = np.array(omegas)
        marks = " ".join(f"{curves[mu_spin][i]:+6.2f}" for i in [0, 4, 9, 19, 49, 99])
        print(f"  mu_spin={mu_spin:.3f}: ω[t=1,5,10,20,50,100] = [{marks}]")

    ref = curves[0.05]
    print("\n  相对真值(0.05)的 MSE（越大越易区分）：")
    for mu_spin, c in curves.items():
        print(f"    mu_spin={mu_spin:.3f}: MSE={np.mean((c - ref) ** 2):.4f}")


def experiment_4_lateral_coupling():
    """实验 4：固定 mu_spin=0.05，扫描 mu_lat，评估 lateralFriction 对衰减曲线的耦合。"""
    print("\n" + "=" * 72)
    print("实验 4：mu_lat 扫描（mu_spin 固定 0.05），评估 lateralFriction 耦合")
    print("=" * 72)
    omega0 = 10.0
    steps = 100
    curves = {}
    for mu_lat in [0.4, 0.6, 0.8, 1.0]:
        client, plane_id, cube_id = _new_world()
        p.changeDynamics(cube_id, -1, lateralFriction=mu_lat, spinningFriction=0.05, rollingFriction=0.0)
        for _ in range(80):
            p.stepSimulation()
        p.resetBaseVelocity(cube_id, [0, 0, 0], [0, 0, omega0])
        omegas = []
        for _ in range(steps):
            p.stepSimulation()
            _lin, ang = p.getBaseVelocity(cube_id)
            omegas.append(ang[2])
        p.disconnect(client)
        curves[mu_lat] = np.array(omegas)
        marks = " ".join(f"{curves[mu_lat][i]:+6.2f}" for i in [0, 4, 9, 19, 49, 99])
        print(f"  mu_lat={mu_lat:.2f}: ω[t=1,5,10,20,50,100] = [{marks}]")

    ref = curves[0.8]
    print("\n  相对 mu_lat=0.8 的 MSE（对比实验 3 的 mu_spin 区分度）：")
    for mu_lat, c in curves.items():
        print(f"    mu_lat={mu_lat:.2f}: MSE={np.mean((c - ref) ** 2):.4f}")


def experiment_5_stage_S_in_simulator():
    """实验 5：在 RobotSimulator（含手臂）里跑 Stage S 流程，验证集成后的信号。"""
    print("\n" + "=" * 72)
    print("实验 5：RobotSimulator 环境下的 Stage S 平面扭转衰减信号")
    print("=" * 72)

    sim = RobotSimulator(gui=False)
    sim.connect()

    base_params = {
        "mass": 0.15, "mu_lat": 0.80, "mu_spin": 0.05,
        "k_n": 5000.0, "c_n": 50.0,
        "com_dx": 0.005, "com_dy": -0.005, "com_dz": 0.002,
        "joint_damp": 1.5, "sys_delay": 0.025,
    }

    for mu_spin in [0.005, 0.02, 0.05, 0.08, 0.12]:
        params = dict(base_params, mu_spin=mu_spin)
        sim.set_params(params)
        # Stage S 流程：立方体平放平面，手臂 [0]*7 远离，给绕 z 轴初始角速度
        sim.reset_to_state(cube_pos=(0.5, 0.0, 0.025), arm_j=[0.0] * 7, finger_pos=0.04)
        sim.action_buffer.clear()
        for _ in range(sim.delay_steps):
            sim.action_buffer.append(([0.0] * 7, 0.04, 5.0))
        for _ in range(60):  # settle
            sim._apply_action([0.0] * 7, finger_pos=0.04, finger_force=5.0)
            p.stepSimulation()
        p.resetBaseVelocity(sim.cube_id, [0, 0, 0], [0, 0, 10.0])
        omegas = []
        for _ in range(50):
            sim._apply_action([0.0] * 7, finger_pos=0.04, finger_force=5.0)
            p.stepSimulation()
            _lin, ang = p.getBaseVelocity(sim.cube_id)
            omegas.append(ang[2])
        omegas = np.array(omegas)
        print(f"  mu_spin={mu_spin:.3f}: ω[t=1,3,5,10,20] = "
              f"[{omegas[0]:+6.2f} {omegas[2]:+6.2f} {omegas[4]:+6.2f} "
              f"{omegas[9]:+6.2f} {omegas[19]:+6.2f}]")

    sim.disconnect()


if __name__ == "__main__":
    experiment_1_torsional_decay()
    experiment_2_grasp_torsion()
    experiment_3_decay_curve_sensitivity()
    experiment_4_lateral_coupling()
    experiment_5_stage_S_in_simulator()
