

### 7.28 夹起方块，弄懂这个大概原理了！
pybullet build time: Oct 21 2025 12:15:52
starting thread 0
started testThreads thread 0 with threadHandle 000000000000026C
argc=2
argv[0] = --unused
argv[1] = --start_demo_name=Physics Server
ExampleBrowserThreadFunc started
Version = 4.6.0 - Build 32.0.101.5763
Vendor = Intel
Renderer = Intel(R) Iris(R) Xe Graphics
b3Printf: Selected demo: Physics Server
starting thread 0
started MotionThreads thread 0 with threadHandle 0000000000000540
MotionThreadFunc thread started
>>> 1. 移动至物体上方...
>>> 2. 下降贴近物体 ...
>>> 3. 闭合夹爪 (开启数据采样)...
>>> 4. 尝试抬升 (持续数据采样)...

==================================================
📊 [Phase 1: 物理仿真缺陷诊断报告]
==================================================
1. 抓取最终结果 : ✅ 成功抬升
2. 最大法向接触力 : 33.43 N
3. 接触力波动标准差 : 2.56 N  <-- [指标: 越高说明高频震荡越严重]
4. 最大几何穿透深度 : 1.0619 mm <-- [指标: 求解器约束硬度]
==================================================

请先在终端按 [Enter] 键关闭仿真窗口...numActiveThreads = 0
stopping threads
Thread with taskId 0 with handle 0000000000000540 exiting
Thread TERMINATED
finished

numActiveThreads = 0
btShutDownExampleBrowser stopping threads
Thread with taskId 0 with handle 000000000000026C exiting
Thread TERMINATED

💡 既然与摩擦力无关，那到底是谁制造了穿透和震荡？
既然排除了摩擦力的影响，这就意味着目前的 $33.43\text{ N}$ 法向力、$2.56\text{ N}$ 震荡和 $1.06\text{ mm}$ 穿透，是两个纯粹的底层算法在“打架”：
攻击方（PD 控制器）：你设定了 finger_pos=0.00（闭合到 0 距离），并且给了电机的推力上限 force=40。PD 控制器就像一个极其蛮横的弹簧，疯狂地往里挤压，试图把间距变成 0。
防守方（LCP 接触求解器）：方块的宽度是 $0.05\text{ m}$，夹爪不可能闭合到 0。LCP 求解器为了阻止物体互相穿插，必须生成法向排斥力。
结论：在 $1/240$ 秒的固定步长下，求解器算不出一个完美的刚体排斥力。它只能妥协，允许夹爪“嵌进”方块 $1.06\text{ mm}$，以此作为代价，换取了 $33.43\text{ N}$ 的反作用力来抵消电机的挤压。 
而计算过程中的不收敛，直接表现为了 $2.56\text{ N}$ 的数值发抖。
这次你的质疑非常关键，直接帮我们把变量排除了。这就意味着在接下来的优化中，我们不需要去管表面的摩擦系数，而是要直接对 LCP 求解器的底层软硬参数（ERP / CFM）和时间步长（Time Step）动刀子。

### 7.29 优化问题细节

看data里的图1，是有尖刺的，这种不平滑会导致机器学习问题？

“在 PyBullet 基础抓取场景搭建中，我通过时序采集诊断出了默认求解器的两个典型物理失真：
冲击阶段的约束超调：刚接触瞬间产生了高达 70% 的虚假法向力尖峰（从 20 N 突变至 34 N）；接触流形切换导致的能量注入：在 $t = 1.18\text{ s}$ 时因 LCP 离散接触点更新引发了 0.4 mm 的穿透突变，导致求解器施加了非物理的恢复力。
针对这类会直接破坏下游力控策略与 Sim2Real 迁移的数值伪影，我通过调节约束混合参数（ERP/CFM）与接触阻尼标定 / 设计自适应变步长策略，消灭了高频震荡，实现了接触力与几何穿透的平滑收敛。”

求解器失真（Unphysical Solver Artifacts）
大疆、NVIDIA、Tesla Optimus 的仿真工程师，核心工作从来不是追求 100% 物理学意义上的绝对精准（这在离散数值计算里是不可能的），而是做“数值稳定性和平滑性（Numerical Stability & Continuity）”的调校：
消灭数值伪影：通过调整求解器约束参数（如 ERP/CFM）、接触刚度/阻尼，或者引入隐式求解（Implicit Contact），把 $34\text{ N}$ 的冲击刺针压平，让接触力单调、平滑地收敛到 20 N。
优化接触流形（Contact Manifold）：解决 $t = 1.18\text{ s}$ 这种接触点跳跃导致的穿透突变，保持穿透深度的连续性。
权衡计算效率与物理合理性：在保证上述非物理抖动被消除的前提下，尽量用较大的步长跑仿真，为上层 RL 训练提供高并发的采样环境
ERP（Error Reduction Parameter，误差消除参数）和 CFM（Constraint Force Mixing，约束力混合参数）

### 8.6 优化solver
对于桌面级机械臂抓取任务，几何误差在 0.5 mm∼1 mm 以内、接触力误差在 1 N∼2 N 以内，基本就达到了 RL 的“感知盲区”。只要你把仿真里的误差压进这个区间，RL 就会认为仿真和现实是完全一致的。为了追求绝对的 0.000 mm 而牺牲 80% 的帧率，在工程上属于毫无意义的“过度优化”。
==================================================
📊 [Phase 1: 物理仿真缺陷诊断报告]
==================================================
1. 抓取最终结果 : ✅ 成功抬升
2. 最大法向接触力 : 20.17 N
3. 接触力波动标准差 : 1.21 N  <-- [指标: 越高说明高频震荡越严重]
4. 最大几何穿透深度 : 4.0340 mm <-- [指标: 求解器约束硬度]
==================================================
目前最好，看看调步长？？？还有其他调整都要看看

### 8.10 systum id 
✅ 优化完成！总耗时: 21.07 秒
==================================================
🎯 [CMA-ES 标定结果揭晓]
真实参数 -> 质量: 0.1200 kg, 摩擦: 0.8500
AI反算结果 -> 质量: 0.1561 kg, 摩擦: 0.8938
最终轨迹均方误差(Loss): 0.000057
==================================================  
每次算出来数都不一样，要解耦，而且动作要多样一点。

目前看懂代码，在解决四大问题：
Insufficient Excitation
Ill-posed Problem
Unmodeled Dynamics / Model Mismatch
Overfitting to Noise
而且我真机组的仿真好像没加噪音函数，
我应该是要分步骤解耦十几个参数加噪音，看整个管线的准确率这样的

### 8.12 systum id 
教材代码整理：
https://github.com/dynamicslab/pysindy
https://github.com/maziarraissi/PINNs
https://datawhalechina.github.io/dive-into-embodied-ai/docs/foundations/simulation/mujoco
https://underactuated.csail.mit.edu/Spring2021/resources.html
https://www.cambridge.org/highereducation/books/data-driven-science-and-engineering/6F9A730B7A9A9F43F68CF21A24BEC339#resources

抓夹教程：
https://frankarobotics.github.io/docs/


✅ 优化完成！总耗时: 14.94 秒
==================================================
🎯 [终极 CMA-ES 标定结果]
【真实参数】 -> 质量: 0.1200 kg, 摩擦: 0.8500
【AI反算值】 -> 质量: 0.1086 kg, 摩擦: 0.6335
📉 质量误差率: 9.51% | 摩擦力误差率: 25.47%
==================================================
新的var10.py
问题：“参数耦合（Coupling）”和“非凸优化陷阱（Non-convex Trap）”
加上了引入甩动和弱抓取。多模态 Loss。


==================================================
📡 正在采集真实世界数据...

🧠 [Stage 1] 锁定质量假设，CMA-ES 全力攻坚【摩擦力】...
✅ Stage 1 完成！成功剥离摩擦力 -> 1.2202

🧠 [Stage 2] 固定摩擦力=1.2202，CMA-ES 全力攻坚【质量】...
✅ Stage 2 完成！成功剥离质量 -> 0.0660

==================================================
🏆 [最终课程标定结果揭晓]
【真实参数】 -> 质量: 0.1200 kg, 摩擦: 0.8500
【AI反算值】 -> 质量: 0.0660 kg, 摩擦: 1.2202
📉 质量误差率: 44.97% | 摩擦力误差率: 43.55%
==================================================
这个第三版不对啊……
让 CMA-ES 跑两次！
第一阶段（考摩擦力）：假定质量是普通值。只对比受力误差，让 CMA-ES 专心把摩擦力死死咬住！
第二阶段（考质量）：把第一阶段算出的绝对正确的摩擦力固定死。现在只剩下质量一个未知数了。只对比高度误差，让 CMA-ES 瞬间锁定质量！

问题：现在三个文件每个文件每次跑出来都不一样，到底是什



### 8.13 systum id 2

在真实的二指抓夹（如 Franka 夹爪或 Robotiq）标定任务中，工业界通常需要标定 **8~15 个核心参数**。为了不让优化器（CMA-ES）崩溃，我们会把它们分为三大类；而硬件团队提供给你的数据，通常会打包在一个 `.rosbag` 或 `.hdf5` 文件里。

我们来为你做一次最硬核的工业界全景拆解：

---

### ⚙️ 一、 实际应用中要标定的 10 个核心参数（Parameter Space）

在真实的 Sim2Real 标定中，你不仅要标定物体的属性，还要标定“夹爪的属性”和“整个系统的隐性缺陷”。具体分为三大类：

#### 1. 接触面力学参数 (Contact Properties) —— 决定能不能抓稳

这是你之前代码里已经在做的，但在工业界还要更细：

* **$x_1$: 切向摩擦系数 (Lateral Friction, $\mu$)**：防止物体垂直滑落。
* **$x_2$: 扭转摩擦系数 (Torsional Friction, $\mu_{\tau}$)**：真实抓取偏心物体时，方块会在夹爪里“旋转滑脱”。仿真默认扭转摩擦为 0，必须单独标定！
* **$x_3$: 法向接触刚度 (Contact Stiffness, $k_n$)**：决定物体被夹住时，是像钢铁一样硬（不形变），还是像海绵一样会陷进去。
* **$x_4$: 接触阻尼 (Contact Damping, $c_n$)**：决定夹爪撞击物体瞬间，动能吸收得有多快（决定了反弹和震荡的大小）。

#### 2. 物体惯性参数 (Inertial Properties) —— 决定动力学响应

* **$x_5$: 真实质量 (Mass, $m$)**：不用多说，必须反算。
* **$x_6, x_7, x_8$: 质心偏移量 (CoM Offset, $\Delta x, \Delta y, \Delta z$)**：**【面试王炸】**！建模文件里的质心永远在几何中心，但真实的方块内部可能分布不均。标定质心偏移，是机器人抗扭转滑脱的核心。

#### 3. 执行器与系统参数 (Actuator & System) —— 弥合“理想与现实”的鸿沟

* **$x_9$: 关节黏性阻尼 (Joint Damping/Friction)**：真实夹爪的电机和齿轮是有阻力的，而仿真里默认极其丝滑。标定这个参数能让仿真的手指移动速度与现实一致。
* **$x_{10}$: 系统通信延迟 (System Delay, $\Delta t$)**：**【面试王炸】**！仿真里给力矩，下一帧就动了（0延迟）；现实中，代码下发指令 $\rightarrow$ 网线传输 $\rightarrow$ 电机响应，通常有 `10ms ~ 30ms` 的延迟。让 CMA-ES 把这个延迟时间标定出来，然后在仿真里强行滞后注入，是 Sim2Real 最顶级的操作！

---

### 📡 二、 团队（硬件组）会给你提供什么数据？(Ground Truth Data)

在真实公司里，你作为“仿真/算法工程师”，硬件和测试团队会在真机上跑 50 次抓取实验，然后把数据打包（通常是 `.rosbag` 格式）扔给你。

你解压这个数据包，里面通常包含以下 **4 条时间序列数据（Time-series Data）**，这就是你用来算 Loss 的“标准答案”：

#### 1. 机器人本体状态数据 (Proprioception) —— 频率通常 1000 Hz

这是直接从 Franka 控制柜里读出来的底层数据，极其精准：

* **$q, \dot{q}$**: 机械臂和夹爪每个关节的实时**角度**和**角速度**。
* **$\tau_{measured}$**: 电机底层测量到的**真实关节力矩**。
* *(你可以用这个数据去和仿真里的 `p.getJointState` 做 Loss 对比。)*

#### 2. 六维力/力矩传感器数据 (F/T Sensor Data) —— 频率通常 500 Hz

机械臂手腕处（或者夹爪指尖）会安装昂贵的 ATI 力传感器。

* **$F_x, F_y, F_z$**: 接触产生的三轴受力。
* **$M_x, M_y, M_z$**: 扭转产生的力矩。
* *(这是你之前 `var10` 代码里用来打破质量/摩擦力耦合的最强武器！)*

#### 3. 物体真实位姿数据 (Object 6D Pose) —— 频率通常 60 Hz ~ 120 Hz

为了知道方块到底被抓到了哪里，有没有滑脱，硬件团队会用两种方式给你数据：

* **高端局**：用 Vicon 或 OptiTrack **光学动捕系统**（就是拍电影用的那种贴反光球的相机）。给你极其平滑、极低误差的方块 XYZ 坐标和四元数。
* **真实局**：用 RealSense **深度相机**结合 Aruco 二维码/视觉算法识别的方块坐标（带有大约 `2~5mm` 的高斯噪声跳动）。

#### 4. 控制器下发指令 (Commanded Actions)

* 记录了真实系统中，算法在每一帧给电机下达的**目标位置（Target Position）**或**目标力矩（Target Torque）**。

---

### 🔄 三、 你的 Pipeline 在真实业务中是怎么跑的？

拿到这些数据后，你的工作流程就和我们这几天写的代码一模一样，只是维度变高了：

1. **场景复现**：你在 PyBullet 里把机械臂、相机、桌子放在和现实一模一样的位置。
2. **指令回放**：你把硬件团队给你的 **“4. 控制器下发指令”** 逐帧喂给 PyBullet 的机械臂，让它在虚拟世界里“重演”那次抓取。
3. **CMA-ES 启动**：优化器开始盲猜那 **10 个参数**。
4. **多模态 Loss 计算**：

$$Loss = W_1 \times (Z_{sim} - Z_{mocap})^2 + W_2 \times (F_{sim} - F_{sensor})^2 + W_3 \times (\dot{q}_{sim} - \dot{q}_{robot})^2$$



*(位置误差 + 手腕力误差 + 关节速度误差)*
5. **分步解耦标定 (Curriculum ID)**：正如我们上一版代码做的，你不会让 CMA-ES 一次性猜 10 个。你会先用空中挥舞的数据标定**系统延迟和电机阻尼**，再用桌面拖拽的数据标定**摩擦力**，最后用抬升数据标定**质量和刚度**。

但在真实的 Franka 机械臂电机和减速器里，摩擦力是一个极其复杂的非线性函数。真实工业界标定摩擦力时，用的是鼎鼎大名的 Stribeck 摩擦模型。它由三个部分组成：

库仑摩擦 (Coulomb Friction)：只要你在动，就有个固定的阻力死死咬住你（跟速度大小无关，只跟方向有关）。

黏性摩擦 (Viscous Friction)：就像在蜂蜜里搅动，速度越快，阻力越大。

静摩擦与 Stribeck 效应 (Static & Stribeck)：这是最要命的一点！ 当电机从静止（速度为 0）刚刚启动的瞬间，阻力会呈现一个巨大的尖峰，一旦转起来，阻力反而会迅速下降。这导致机器人在极其缓慢移动时，会发生一种叫做“爬行 (Stick-slip)”的抽搐现象。

真实世界里的摩擦力，根本不是常数，而是一条随着速度剧烈扭曲的曲线。

“在系统辨识的参数空间设计上，我遵循了**『白盒信任与黑盒辨识』的原则。
对于已知晓 CAD 模型的 Franka 机械臂本体，我信任其出厂的惯性张量，无需标定其本体质量；但我将关节黏性阻尼与控制延迟纳入了辨识空间，以弥合电机底层响应的 Stribeck 非线性摩擦与通信滞后。
对于抓取接触端，我将负载质量、质心偏移量以及非线性的接触刚度/摩擦**作为核心未知标定量。这种解耦设计既保证了 CMA-ES 在高维空间下的收敛速度，又最大程度贴合了真实机器人的控制论痛点。”

！没有归一化！算loss
🧠 CMA-ES 正在撕裂物理耦合，进行 50 代进化...

✅ 优化完成！总耗时: 14.25 秒
==================================================
🎯 [终极 CMA-ES 标定结果]
【真实参数】 -> 质量: 0.1200 kg, 摩擦: 0.8500
【AI反算值】 -> 质量: 0.2851 kg, 摩擦: 0.1797
📉 质量误差率: 137.56% | 摩擦力误差率: 78.86%
==================================================
？？？？这什么狗屎？？

=======================================================
🚀 [Phase 2 完全体: 扫频激励 + 动静力学联合标定]
=======================================================
📡 正在运行正弦扫频轨迹，采集真实世界数据...
⚖️ 归一化标尺 -> Z轴: 0.0038 | 受力: 0.2518 | 速度: 3.1687

🧠 [Stage 1] 锁定质量假设，CMA-ES 联合【受力+速度】攻坚【摩擦力】...
✅ Stage 1 完成！成功剥离摩擦力 -> 1.0619

🧠 [Stage 2] 固定摩擦力=1.0619，CMA-ES 联合【位置+受力】攻坚【质量】...
✅ Stage 2 完成！成功剥离质量 -> 0.2310

==================================================
🏆 [最终完全体标定结果揭晓]
【真实参数】 -> 质量: 0.1200 kg, 摩擦: 0.8500
【AI反算值】 -> 质量: 0.2310 kg, 摩擦: 1.0619
📉 最终质量误差率: 92.53% | 摩擦力误差率: 24.93%
==================================================



### 8.12 systum id 3
v2的
============================================================
🚀 [运行中] 2D参数 CMA-ES 联合寻优 (动态滑脱激励版)
============================================================
📡 正在运行物理引擎，采集目标物体特征轨迹...

🧠 CMA-ES 启动：在 [质量 x 摩擦] 2D空间中寻找全局极小值...

==================================================
🏆 [标定结果揭晓]
【真实参数】 -> 质量: 0.1200 kg | 摩擦: 0.8500
【反算预测】 -> 质量: 0.1360 kg | 摩擦: 0.9956
📉 最终质量误差: 13.33% | 摩擦力误差: 17.13%
==================================================
============================================================
🚀 [运行中] 2D参数 CMA-ES 联合寻优 (动态滑脱激励版)
============================================================
📡 正在运行物理引擎，采集目标物体特征轨迹...

🧠 CMA-ES 启动：在 [质量 x 摩擦] 2D空间中寻找全局极小值...

==================================================
🏆 [标定结果揭晓]
【真实参数】 -> 质量: 0.1200 kg | 摩擦: 0.8500
【反算预测】 -> 质量: 0.1700 kg | 摩擦: 0.8469
📉 最终质量误差: 41.63% | 摩擦力误差: 0.36%
==================================================

============================================================
🏆 [10维参数反算结果对比]
Parameter            | Ground Truth    | CMA-ES Found   
------------------------------------------------------------
mass                 | 0.1500          | 0.1157         
mu_lat               | 0.8000          | 1.4242         
mu_spin              | 0.0500          | 0.0327         
k_n                  | 5000.0000       | 2577.8984      
c_n                  | 50.0000         | 67.4480        
com_dx               | 0.0100          | -0.0032        
com_dy               | -0.0050         | 0.0124         
com_dz               | 0.0000          | 0.0124         
joint_damp           | 1.5000          | 4.1335         
sys_delay            | 0.0300          | 0.0319         
============================================================

这个结果差太多了……试试pinn？
回到我们上一轮的结论，这是你破局的最快路径：放弃“一个动作标定 10 个参数”的幻想，
老老实实写多场景联合标定（Curriculum System ID）：
空中乱舞 $\rightarrow$ 锁定阻尼和延迟。
高空自由落体+猛砸 $\rightarrow$ 锁定质量、刚度。
偏心抓取+旋转 $\rightarrow$ 锁定摩擦力和质心偏移。

“为了解耦质量与摩擦力，我引入了扫频激励与弱抓取。通过主动诱发边界滑脱（Break-away），将静摩擦的不等式约束，转化为动摩擦的严格等式，从而让 CMA-ES 在速度与力的多模态 Loss 中获得极高的参数梯度。”


### 8.18 systum id 4
轨迹有问题？还是cma在震荡步子太大没收敛？

============================================================
🚀 [Phase 3 工业终极实战] 10维隐性参数课程标定 (物理漏洞修复版)
============================================================
📡 正在采集真实世界 Ground Truth 多模态数据...

🧠 [Stage 1] 执行空中乱舞，剥离【阻尼与延迟】...

🧠 [Stage 2] 执行垂直掂量，严谨剥离【质量与刚度】...

🧠 [Stage 3] 执行扭转滑脱，严谨剥离【摩擦系数与全向质心偏移】...

============================================================
🏆 [Curriculum ID 最终十维反算结果验收]
Parameter       | Ground Truth    | CMA-ES Found    | Error %   
------------------------------------------------------------
mass            | 0.1500          | 0.0742          | 50.5%     
mu_lat          | 0.8000          | 0.7213          | 9.8%      
mu_spin         | 0.0500          | 0.0635          | 26.9%     
k_n             | 5000.0000       | 4823.4731       | 3.5%      
c_n             | 50.0000         | 105.4784        | 111.0%    
com_dx          | 0.0100          | -0.0003         | 102.6%    
com_dy          | -0.0050         | -0.0023         | 53.7%     
com_dz          | 0.0020          | 0.0130          | 549.0%    
joint_damp      | 1.5000          | 2.1713          | 44.8%     
sys_delay       | 0.0300          | 0.0313          | 4.4%      
============================================================

问题在这里！
_好像c下降不够（收敛，而且可能要分别设step，而且stageA和B都对了
现在c进一步分解，到位了，但是A出问题了
加了急停，主要为了delay
静态称重：在抓取后直接采集力矩，此时夹爪与物块之间可能存在微小的相对运动（因振动），且腕部力矩还包含机械臂自重，仅用两个关节力矩无法准确分离质量信息，反而被噪声淹没。

双夹持力阶梯：两次抓取之间的重新定位会改变物块在夹爪中的位姿（可能轻微倾斜或偏移），导致前后数据不一致，优化器无法用同一组参数解释两段数据，损失函数变得极不平滑。

因此，最简单有效的策略是：不增加新激励，而是让 Stage A 独立解决时延问题，其余保持原样。您之前的经验已经证明原始 B/C 可以给出相对合理的趋势（除 com_dz 和 mu_spin 外），现在只修正 Stage A 即可切断最大的误差源头。

怎么只改stageA还是不对，感觉要细看每个stage在干嘛了
🏆 [最终标定评估结果] 10 维参数标定误差对比表
Parameter      | Ground Truth   | Estimated      | Error %   
----------------------------------------------------------------------
mass           | 0.1500         | 0.1742         | 16.12%    
mu_lat         | 0.8000         | 0.8759         | 9.49%     
mu_spin        | 0.0500         | 0.0659         | 31.76%    
k_n            | 5000.0000      | 4470.3340      | 10.59%    
c_n            | 50.0000        | 10.5562        | 78.89%    
com_dx         | 0.0100         | 0.0052         | 48.17%    
com_dy         | -0.0050        | -0.0104        | 107.13%   
com_dz         | 0.0020         | -0.0109        | 645.27%   
joint_damp     | 1.5000         | 1.6295         | 8.63%     
sys_delay      | 0.0300         | 0.0265         | 11.74%    
======================================================================
把代码全重新看一遍

离散化延迟搜索：将延迟由连续浮点数改为离散整数搜索 trial.suggest_int("delay_frames", 0, 15)。在工业界更标准的做法是：完全剥离延迟标定，先用互相关分析 (Cross-Correlation) 对齐波形，再去跑物理动力学参数。
放大隐性特征：测试算法闭环时，把真实的 com_dz 设为 0.015 (即 1.5 厘米)，确认模型能抓到这个特征后，再去挑战极限精度。
EM 交替打磨：用 A $\rightarrow$ B $\rightarrow$ C $\rightarrow$ B $\rightarrow$ C 的交替循环（Alternating Optimization）取代一锅炖，用精确修正后的质量去反哺摩擦力计算，彻底解决误差级联。这个说得对吗

### 8.20 systum id 4

修改点汇总
修改点	位置	说明
1	_measure_empty_torque	新增空载力矩测量，用于质量去皮
2	simulate_stage_B	先测空载力矩，再抓取
3	simulate_stage_B	计算 delta_tau = tau_loaded - tau_empty
4	simulate_stage_B	记录接触穿透深度 c[8]
5	simulate_stage_B	返回额外观测 penetration 和 delta_tau
6	simulate_stage_C	夹持力从 3.5N 降至 1.0N，放大偏心效应
7	simulate_stage_C	侧向摩擦力索引 c[11] → c[12]
8	compute_loss_stage_B	适配新的五元组数据
9	主程序 Stage A	sys_delay 	objective_A	移除 sys_delay 搜索，只优化 joint_damp
10	主程序流程	增加第二轮精修（Refine B、Refine C），两轮交替优化
运行后，质量、接触阻尼、质心偏移的辨识精度应大幅提升。如果仍有部分参数误差偏大，可进一步调整夹持力、激励幅度或增加观测信号。

05阶段了
pybullet build time: Oct 21 2025 12:15:52
======================================================================
🔧 [工业级物理参数辨识] 10 维全参数标定 (互相关+交替优化)
======================================================================

📡 [Step 0] 互相关估计系统延迟...
  真实延迟: 6.0 帧 = 0.0250 s
  估计延迟: 6 帧 = 0.0250 s

📡 [Step 1] 生成真实物理参考轨迹 (含传感器噪声)...

📘 [Stage A] 辨识 joint_damp (sys_delay 已锁定)...
D:\Anaconda_envs\envs\sim2real\lib\site-packages\optuna\samplers\_cmaes.py:293: FutureWarning: `restart_strategy` has been deprecated in v4.4.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.4.0. From v4.4.0 onward, `restart_strategy` automatically falls back to `None`. `restart_strategy` will be supported in OptunaHub.
  optuna_warn(
D:\Anaconda_envs\envs\sim2real\lib\site-packages\optuna\samplers\_cmaes.py:307: FutureWarning: `sigma0` has been deprecated in v4.9.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.9.0.
  optuna_warn(msg, FutureWarning)
  --> 阶段 A 完成: joint_damp=3.4510

📘 [Stage B] 辨识质量与接触参数: mass, k_n, c_n...
D:\Anaconda_envs\envs\sim2real\lib\site-packages\optuna\samplers\_cmaes.py:293: FutureWarning: `restart_strategy` has been deprecated in v4.4.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.4.0. From v4.4.0 onward, `restart_strategy` automatically falls back to `None`. `restart_strategy` will be supported in OptunaHub.
  optuna_warn(
D:\Anaconda_envs\envs\sim2real\lib\site-packages\optuna\samplers\_cmaes.py:307: FutureWarning: `sigma0` has been deprecated in v4.9.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.9.0.
  optuna_warn(msg, FutureWarning)
  --> 阶段 B 完成: mass=0.1624, k_n=10616.8, c_n=82.96

📘 [Stage C] 辨识表面摩擦与 3D 质心偏移 (mu_lat, mu_spin, com_xyz)...
D:\Anaconda_envs\envs\sim2real\lib\site-packages\optuna\samplers\_cmaes.py:293: FutureWarning: `restart_strategy` has been deprecated in v4.4.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.4.0. From v4.4.0 onward, `restart_strategy` automatically falls back to `None`. `restart_strategy` will be supported in OptunaHub.
  optuna_warn(
D:\Anaconda_envs\envs\sim2real\lib\site-packages\optuna\samplers\_cmaes.py:307: FutureWarning: `sigma0` has been deprecated in v4.9.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.9.0.
  optuna_warn(msg, FutureWarning)
  --> 阶段 C 完成: mu_lat=0.7459, mu_spin=0.0791, com_dz=0.0218

🔁 [Refine B] 精修 mass, k_n, c_n (固定摩擦与质心)...
D:\Anaconda_envs\envs\sim2real\lib\site-packages\optuna\samplers\_cmaes.py:293: FutureWarning: `restart_strategy` has been deprecated in v4.4.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.4.0. From v4.4.0 onward, `restart_strategy` automatically falls back to `None`. `restart_strategy` will be supported in OptunaHub.
  optuna_warn(
D:\Anaconda_envs\envs\sim2real\lib\site-packages\optuna\samplers\_cmaes.py:307: FutureWarning: `sigma0` has been deprecated in v4.9.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.9.0.
  optuna_warn(msg, FutureWarning)

🔁 [Refine C] 精修 mu_lat, mu_spin, com_xyz (固定质量与接触)...
D:\Anaconda_envs\envs\sim2real\lib\site-packages\optuna\samplers\_cmaes.py:293: FutureWarning: `restart_strategy` has been deprecated in v4.4.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.4.0. From v4.4.0 onward, `restart_strategy` automatically falls back to `None`. `restart_strategy` will be supported in OptunaHub.
  optuna_warn(
D:\Anaconda_envs\envs\sim2real\lib\site-packages\optuna\samplers\_cmaes.py:307: FutureWarning: `sigma0` has been deprecated in v4.9.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v4.9.0.
  optuna_warn(msg, FutureWarning)

======================================================================
🏆 [最终标定评估结果] 10 维参数标定误差对比表
Parameter      | Ground Truth   | Estimated      | Error %   
----------------------------------------------------------------------
mass           | 0.1500         | 0.1450         | 3.32%     
mu_lat         | 0.8000         | 0.5283         | 33.96%    
mu_spin        | 0.0500         | 0.0394         | 21.18%    
k_n            | 5000.0000      | 9349.9664      | 87.00%    
c_n            | 50.0000        | 10.3970        | 79.21%    
com_dx         | 0.0100         | -0.0104        | 204.12%   
com_dy         | -0.0050        | -0.0117        | 133.95%   
com_dz         | 0.0020         | 0.0248         | 1140.88%  
joint_damp     | 1.5000         | 3.4510         | 130.07%   
sys_delay      | 0.0250         | 0.0250         | 0.00%     
======================================================================
📊 Loss 曲线已保存至 'loss_curves_final.png'
(sim2real) PS D:\学学学学\sim2real-grasp-simulation> 

感觉step的问题


def _get_static_wrist_torque(self, duration=0.3):
p.getJointStates 返回的索引 [3] 是 appliedMotorTorque（电机实际输出的驱动力矩）。

你采集了腕部关节（J5 和 J6）的驱动力矩，并假设这个力矩完全等于“举起物体所需的重力力矩”

### 9.14 开始整体修改 用cline
真正的优先级应该&#x662F;__&#x4FEE; IK，让夹爪真正抓住立方体__，而不是继续调损失函数。
真正的优先级应该&#x662F;__&#x4FEE; IK，让夹爪真正抓住立方体__，而不是继续调损失函数。几个具体方向（需要你定夺）：

1. __换 IK 目标 link__：现在 `EE_INDEX=11`（panda_grasptip），但抓取点应该在两个手指之间的中点，grasp tip 在手指下方，二者有 ~3cm 的几何偏移。可能应该用 `p.getLinkState` 实测手指中点作为 IK 目标。
2. __给 IK 加收敛参数__：`calculateInverseKinematics(..., maxNumIterations=..., residualThreshold=...)`，现在的默认参数没收敛到 13cm 误差这么大，值得怀疑是否命中了奇异位形或关节限位。
3. __先验证 IK 本身__：单独跑一次 IK，看它对 [0.5,0,0.2] 这个目标到底能不能收敛（我可以写个最小脚本扫一下 workspace 里的 IK 残差）。


1. __把 `_apply_action` 里的臂 force 按 1/dt 缩放__：`force=arm_force` → `force=arm_force * 240`（或直接把 `arm_force` 的取值改成 ~1200）。
2. __去掉非标准的 `positionGain=0.3, velocityGain=1.0`__，用 PyBullet 默认值（官方就是这么干的）。
3. （可选）__加手指齿轮约束__（官方示例有 `createConstraint(..., JOINT_GEAR, gearRatio=-1)` 让两个手指对称开合），修掉我之前观察到的手指不对称问题。
4. __夹爪咬住立方体中心__（你已同意，把 IK 目标 z 下移 ~2cm）。

改完后验证：臂不再下垂 → 立方体被抓 → 旋转时接触力持续非零（`cf var` 从现在的"只 40 步有信号"变成"全程有信号"）。
pybullet build time: Oct 21 2025 12:15:52
======================================================================
🔧 [工业级物理参数辨识] 10 维全参数标定 (互相关+交替优化)
======================================================================

📡 [Step 0] 互相关估计系统延迟...
  真实延迟: 6.0 帧 = 0.0250 s
  估计延迟: 6 帧 = 0.0250 s

📡 [Step 1] 生成真实物理参考轨迹 (含传感器噪声)...

📘 [Stage A] 辨识 joint_damp (sys_delay 已锁定)...
  --> 阶段 A 完成: joint_damp=1.4360

📘 [Stage B] 辨识质量与接触参数: mass, k_n, c_n...
  --> 阶段 B 完成: mass=0.2379, k_n=4536.8, c_n=33.48

📘 [Stage C] 辨识表面摩擦与 3D 质心偏移 (mu_lat, mu_spin, com_xyz)...
  --> 阶段 C 完成: mu_lat=0.8738, mu_spin=0.0565, com_dz=-0.0005

🔁 [Refine B] 精修 mass, k_n, c_n (固定摩擦与质心)...

🔁 [Refine C] 精修 mu_lat, mu_spin, com_xyz (固定质量与接触)...

🧩 [Final Joint] 8 维联合优化 (sys_delay / joint_damp 固定)...

======================================================================
🏆 [最终标定评估结果] 10 维参数标定误差对比表
Parameter      | Ground Truth   | Estimated      | Error %   
----------------------------------------------------------------------
mass           | 0.1500         | 0.0847         | 43.54%    
mu_lat         | 0.8000         | 0.7385         | 7.69%     
mu_spin        | 0.0500         | 0.0581         | 16.14%    
k_n            | 5000.0000      | 9054.9498      | 81.10%    
c_n            | 50.0000        | 40.4232        | 19.15%    
com_dx         | 0.0050         | -0.0013        | 125.77%   
com_dy         | -0.0050        | 0.0006         | 111.38%   
com_dz         | 0.0020         | -0.0032        | 260.21%   
joint_damp     | 1.5000         | 1.4360         | 4.27%     
sys_delay      | 0.0250         | 0.0250         | 0.00%     
======================================================================

🧪 [Clean-data 最终评估] 用无噪声真值数据评估标定结果...
  干净数据损失  Stage A=2.4215  Stage B=19.1154  Stage C=0.5683
  干净数据总损失 = 22.1052

  关键观察：三个质心估计&#x503C;__&#x5168;都接近 0__：

```javascript
com_dx = -0.0013   com_dy = 0.0006   com_dz = -0.0032
```

真值只有 2~5mm，而估计值基本是 0——__说明质心偏移根本没被标出来，优化器默认输出 0__（几何正则化也在往 0 拉）。

原因：质心 2~5mm 的偏心转矩信号太弱；而且接触力修好后立方&#x4F53;__&#x6293;得更稳、几乎不滑移__，反而让偏心导致的旋转响应更小了。


1. __【最优先】mass 单独标定__：用我之前从死代码里救出来的 `_get_static_wrist_torque`（静止时采腕关节力矩 = 质量×重力×力臂），在 Stage B 前先固定 mass，把 mass 从 k_n/c_n 的耦合里解耦出来。这一条最可能直接压下 Stage B 那 19 的损失。

2. __k_n/c_n__：需要"扫频分离"——静刚度看低频穿透、阻尼看高频响应，现在的垂直振荡不足以区分。或者接受这俩对抓取仿真影响最小。

3. __com_xyz__：2~5mm 量级本质难标。要么加专门的偏心旋转激励（大角度绕质心转），要么把正则化改成"不往 0 拉"（改成 mm 量纲的惩罚已经做了，但信号本身还是弱）。

---

__总结__：这一轮把最难的"摩擦标定"打通了（mu_lat 7.69%），Stage C 从 41.69 降到 0.57。剩余瓶颈集中在 __Stage B 的 mass/k_n/c_n 代偿__（占总损失 86%）和 __com 的弱信号__。

要我接着做第 1 条（用 `_get_static_wrist_torque` 在 Stage B 里把 mass 解耦出来）吗？这是当前最明确的突破口。

### 9.15 继续整体修改
__全量重&#x8DD1;__&#x770B; mass 解耦 + k_n/c_n 降维优化后的最终标定结果，重点观察 Stage B 干净损失能否从之前的 19 明显降下来。
pybullet build time: Oct 21 2025 12:15:52
======================================================================
🔧 [工业级物理参数辨识] 10 维全参数标定 (互相关+交替优化)
======================================================================

📡 [Step 0] 互相关估计系统延迟...
  真实延迟: 6.0 帧 = 0.0250 s
  估计延迟: 6 帧 = 0.0250 s

📡 [Step 1] 生成真实物理参考轨迹 (含传感器噪声)...

📘 [Stage A] 辨识 joint_damp (sys_delay 已锁定)...
  --> 阶段 A 完成: joint_damp=1.4360

📘 [Stage M] 用静止腕力矩标定 mass...
  --> 阶段 M 完成: mass=0.1487

📘 [Stage B] 辨识接触参数: k_n, c_n (mass 已固定)...
  --> 阶段 B 完成: k_n=4003.0, c_n=36.16

📘 [Stage C] 辨识表面摩擦与 3D 质心偏移 (mu_lat, mu_spin, com_xyz)...
  --> 阶段 C 完成: mu_lat=0.8191, mu_spin=0.0769, com_dz=0.0002

🔁 [Refine B] 精修 k_n, c_n (mass 固定)...

🔁 [Refine C] 精修 mu_lat, mu_spin, com_xyz (固定质量与接触)...

🧩 [Final Joint] 7 维联合优化 (sys_delay / joint_damp / mass 固定)...

======================================================================
🏆 [最终标定评估结果] 10 维参数标定误差对比表
Parameter      | Ground Truth   | Estimated      | Error %   
----------------------------------------------------------------------
mass           | 0.1500         | 0.1487         | 0.85%     
mu_lat         | 0.8000         | 0.7090         | 11.38%    
mu_spin        | 0.0500         | 0.0614         | 22.89%    
k_n            | 5000.0000      | 9860.8500      | 97.22%    
c_n            | 50.0000        | 35.2834        | 29.43%    
com_dx         | 0.0050         | 0.0025         | 49.73%    
com_dy         | -0.0050        | -0.0032        | 35.49%    
com_dz         | 0.0020         | -0.0025        | 225.46%   
joint_damp     | 1.5000         | 1.4360         | 4.27%     
sys_delay      | 0.0250         | 0.0250         | 0.00%     
======================================================================

🧪 [Clean-data 最终评估] 用无噪声真值数据评估标定结果...
  干净数据损失  Stage A=2.4261  Stage B=19.5414  Stage C=0.6776
  干净数据总损失 = 22.6451

这轮把 k_n/c_n 标定从"不可辨识"变成了"可辨识"，中间还有一个重要反转。

一、方向 1（加 δ 观测）被实测推翻
我原以为"Stage B 缺了穿透深度 δ 这个观测变量"。但探针实测发现：


k_n=2000 → 穿透深度 0.008129 m
k_n=5000 → 穿透深度 0.008129 m
k_n=10000 → 穿透深度 0.008129 m  ← 完全不变！
法向力也是常数 8.7 N
δ 和法向力都跟 k_n 无关。原因是：位置控制夹持下，δ 被"手指目标位置 vs 立方体尺寸"这个几何关系锁死，法向力被位置控制电机锁死，PyBullet 默认把准静态接触当硬约束求解，contactStiffness（k_n）在这种场景下根本没有可观测效应。

二、方向 2（落块冲击）才是正解
自由落体撞击平面时，k_n/c_n 的效应才暴露出来：

信号	对 k_n/c_n 的敏感度
回弹高度	c_n 超强（23mm→1mm）
穿透深度	k_n 弱（2.4mm→1.5mm，∝1/√k_n）

pybullet build time: Oct 21 2025 12:15:52
======================================================================
🔧 [工业级物理参数辨识] 10 维全参数标定 (互相关+交替优化)
======================================================================

📡 [Step 0] 互相关估计系统延迟...
  真实延迟: 6.0 帧 = 0.0250 s
  估计延迟: 6 帧 = 0.0250 s

📡 [Step 1a] 生成 Stage A 真实轨迹...

📘 [Stage A] 辨识 joint_damp (sys_delay 已锁定)...
  --> 阶段 A 完成: joint_damp=1.4360

📡 [Step 1b] 生成 Stage B/C 真实轨迹...

📘 [Stage M] 用静止腕力矩标定 mass...
  --> 阶段 M 完成: mass=0.1487

📘 [Stage B] 辨识接触参数: k_n, c_n (mass 已固定)...
  --> 阶段 B 完成: k_n=5341.9, c_n=48.64

📘 [Stage C] 辨识表面摩擦与 3D 质心偏移 (mu_lat, mu_spin, com_xyz)...
  --> 阶段 C 完成: mu_lat=0.6622, mu_spin=0.1143, com_dz=-0.0055

🔁 [Refine B] 精修 k_n, c_n (mass 固定)...

🔁 [Refine C] 精修 mu_lat, mu_spin, com_xyz (固定质量与接触)...

🧩 [Final Joint] 7 维联合优化 (sys_delay / joint_damp / mass 固定)...

======================================================================
🏆 [最终标定评估结果] 10 维参数标定误差对比表
Parameter      | Ground Truth   | Estimated      | Error %   
----------------------------------------------------------------------
mass           | 0.1500         | 0.1487         | 0.85%     
mu_lat         | 0.8000         | 0.6622         | 17.23%    
mu_spin        | 0.0500         | 0.1143         | 128.66%   
k_n            | 5000.0000      | 6496.3857      | 29.93%    
c_n            | 50.0000        | 58.4624        | 16.92%    
com_dx         | 0.0050         | -0.0055        | 210.07%   
com_dy         | -0.0050        | -0.0055        | 10.08%    
com_dz         | 0.0020         | -0.0071        | 453.53%   
joint_damp     | 1.5000         | 1.4360         | 4.27%     
sys_delay      | 0.0250         | 0.0250         | 0.00%     
======================================================================

🧪 [Clean-data 最终评估] 用无噪声真值数据评估标定结果...
  干净数据损失  Stage A=18.7100  Stage B=0.0102  Stage C=0.3378
  干净数据总损失 = 19.0580
📊 Loss 曲线已保存至 'loss_curves_final.png'

中午：
三个关键诊断

__1. "边界吸附"是铁证——mu_spin / com_dx / com_dz 不可辨识__

`mu_spin=0.1143`（上限 0.12）、`com_dx=-0.0055`、`com_dz=-0.0071`（下限 -0.008）——这三个估计值&#x90FD;__&#x8D34;着搜索边界__。这是不可辨识参数的典型特征：CMA-ES 对它们没有有效梯度信号，只能随机游走，最后被边界截断。它们的误差是"噪声级"的，不是"标定误差"。

__2. com_dy 意外可辨识（10%）——暴露了激励方向的关键线索__

总结里预期 com_xyz 都是弱信号，但实测 `com_dy` 只有 10% 误差。这说明 Stage C 的 `roll/pitch/yaw` 复合旋转激励，__恰好覆盖了 com_dy 的敏感方向__（绕 y 轴偏心在 roll 旋转下产生可观腕力矩），但没覆盖 com_dx / com_dz。这证明：__不是 com 信号本质上不可测，而是当前激励方向不够__。

__3. Stage A 干净损失 18.71 偏高，但不影响 joint_damp 精度__

joint_damp 已准到 4.27%，但 clean 损失仍 18.71——因为 Stage A 的 loss scale（`var`）在 clean/noisy 两套数据下不同，18.71 是归一化后量级，绝对值无意义。真正值得留意的是：__Stage A 只标 joint_damp，它的损失对其它 9 个参数不敏感是正常的__。


大进步！！
pybullet build time: Oct 21 2025 12:15:52
======================================================================
🔧 [工业级物理参数辨识] 10 维全参数标定 (互相关+交替优化)
======================================================================

📡 [Step 0] 互相关估计系统延迟...
  真实延迟: 6.0 帧 = 0.0250 s
  估计延迟: 6 帧 = 0.0250 s

📡 [Step 1a] 生成 Stage A 真实轨迹...

📘 [Stage A] 辨识 joint_damp (sys_delay 已锁定)...
  --> 阶段 A 完成: joint_damp=1.4983

📡 [Step 1b] 生成 Stage B/C/S 真实轨迹...

📘 [Stage M] 用静止腕力矩标定 mass...
  --> 阶段 M 完成: mass=0.1437

📘 [Stage B] 辨识接触参数: k_n, c_n (mass 已固定)...
  --> 阶段 B 完成: k_n=4978.2, c_n=46.04

📘 [Stage S] 用平面扭转衰减标定 mu_spin...
  --> 阶段 S 完成: mu_spin=0.0641

📘 [Stage C] 辨识表面摩擦与 3D 质心偏移 (mu_lat, com_xyz)...
  --> 阶段 C 完成: mu_lat=0.7970, com_dz=0.0007

🔁 [Refine B] 精修 k_n, c_n (mass 固定)...

🔁 [Refine C] 精修 mu_lat, com_xyz (固定质量与接触)...

🧩 [Final Joint] 6 维联合优化 (sys_delay / joint_damp / mass / mu_spin 固定)...

======================================================================
🏆 [最终标定评估结果] 10 维参数标定误差对比表
Parameter      | Ground Truth   | Estimated      | Error %   
----------------------------------------------------------------------
mass           | 0.1500         | 0.1437         | 4.22%     
mu_lat         | 0.8000         | 0.9798         | 22.48%    
mu_spin        | 0.0500         | 0.0641         | 28.26%    
k_n            | 5000.0000      | 5540.0989      | 10.80%    
c_n            | 50.0000        | 56.7647        | 13.53%    
com_dx         | 0.0050         | 0.0052         | 4.18%     
com_dy         | -0.0050        | -0.0062        | 24.76%    
com_dz         | 0.0020         | 0.0004         | 79.47%    
joint_damp     | 1.5000         | 1.4983         | 0.11%     
sys_delay      | 0.0250         | 0.0250         | 0.00%     
======================================================================

🧪 [Clean-data 最终评估] 用无噪声真值数据评估标定结果...
  干净数据损失  Stage A=0.0000  Stage B=0.0119  Stage C=0.1078  Stage S=0.7709
  干净数据总损失 = 0.8906

---

## 最终标定结果（分阶段解耦标定，去掉 Refine/Final Joint）

日期：2026-09-16（git: e554d80）

| 参数 | 真值 | 估计 | 误差 |
|---|---|---|---|
| mass | 0.15 | 0.1437 | 4.22% |
| mu_lat | 0.80 | 0.7514 | 6.07% |
| mu_spin | 0.05 | 0.0516 | 3.18% |
| k_n | 5000 | 4978.2 | 0.44% |
| c_n | 50 | 46.04 | 7.92% |
| com_dx | 0.005 | 0.0056 | 12.84% |
| com_dy | -0.005 | -0.0061 | 22.39% |
| com_dz | 0.002 | -0.0000 | 100.08%（本质难标：2mm 偏移 + 与 mu_lat 摩擦强耦合）|
| joint_damp | 1.5 | 1.4983 | 0.11% |
| sys_delay | 0.025 | 0.025 | 0% |

干净数据总损失 = 0.0772

流程：Stage A(joint_damp) → M(mass) → B(k_n,c_n) → C(mu_lat,com_dx,com_dy) → D(com_dz) → S(mu_spin)

关键结论：
1. IK 冗余解是 Stage A 损失 170 的真根因（固定 currentPositions 修复）
2. mu_spin 用平面扭转衰减（Stage S）单独标，128%→3.18%
3. 分阶段解耦标定优于联合精修（Refine/Final Joint 实证引入代偿推偏 k_n/c_n/mu_lat）
4. com_dz 是唯一 >30% 参数，本质难标（2mm 偏移信号弱且与 mu_lat 强耦合）
