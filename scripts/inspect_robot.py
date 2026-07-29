import pybullet as p
import pybullet_data

p.connect(p.DIRECT)  # 无 GUI 模式快速读取
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

print("\n" + "="*60)
print(f"{'Index':<6} | {'Joint Name':<25} | {'Link Name':<25}")
print("="*60)

for i in range(p.getNumJoints(robot_id)):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode('utf-8')
    link_name = info[12].decode('utf-8')
    print(f"{i:<6} | {joint_name:<25} | {link_name:<25}")

print("="*60 + "\n")
p.disconnect()