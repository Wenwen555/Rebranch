# KinDER 接入与任务开发

本项目使用固定版本 KinDER 的原生对象 MJCF/资产，在 PiPER-X RobotAPI 上构建新任务。官方 KinDER 环境也保留在独立环境中。PiPER-X 变体不是原 KinDER/TidyBot 实验结果。

## 源码与环境

- 上游：<https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden>
- 本地源码：`third_party/kindergarden`，detached commit `5b2dbac482142b879b48449b3fa2af63f641aed3`。
- Python：`.venv-kinder/bin/python`（CPython 3.12.15）。
- 依赖：`requirements/kinder-dynamic3d.lock`；只安装官方 `requirements/dynamic3d.txt` 对应依赖和 ReBranch 所需依赖，没有安装全部后端/开发测试套件。
- 两个 editable 包：此项目及 `third_party/kindergarden`。
- 原 `.venv` 独立保留。新任务使用 `.venv-kinder`，避免 KinDER 的 `mujoco~=3.3.0` 改变旧实验环境。
- 当前渲染选择：`MUJOCO_GL=egl`。本机有 NVIDIA/Mesa EGL，系统库清单没有 OSMesa。

KinDER 的自动注册原来在无 DISPLAY 时无条件覆盖渲染选择为 OSMesa。本地只修改其入口条件，使显式设置的 MUJOCO_GL 生效；未设置时仍沿用上游选择。补丁保存在 `requirements/kinder-egl.patch`，不是隐藏的上游版本变化。

## 重建环境

以下命令用于需要重建的工作目录，本轮仅完成源码/依赖安装，没有运行环境示例或任务回合。

```bash
# 源码首次获取；已有 checkout 不重复 clone。
git clone https://github.com/Princeton-Robot-Planning-and-Learning/kindergarden third_party/kindergarden
git -C third_party/kindergarden checkout --detach 5b2dbac482142b879b48449b3fa2af63f641aed3
git -C third_party/kindergarden apply ../../requirements/kinder-egl.patch
uv venv --python 3.12 .venv-kinder
uv pip install --python .venv-kinder/bin/python -r requirements/kinder-dynamic3d.lock
uv pip install --python .venv-kinder/bin/python --no-deps -e third_party/kindergarden -e .
```

使用该环境时显式设置 `MUJOCO_GL=egl`。官方 Dynamic3D 提供 `scene_bg=False` 的简单背景，可避免任务设计依赖大场景背景；`DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1` 禁止首次创建环境时隐式下载背景资产。需要真实背景时再显式准备 MimicLabs 资产。本轮没有下载这些场景包。

## 复用层与本次任务

- KinDER 的 `GeneratedSeesaw.xml_element` 与 `get_assets()` 提供支座/横梁/铰链和 mesh；`Cube` 提供物块。
- 项目接入层：`rebranch/sim/kinder_scene.py`，负责把本次用到的原生对象及资产接到 PiPER-X 场景，不复制整个上游对象类、不重写机械臂控制器。
- 任务：`rebranch/tasks/hidden_mass_balance_beam.py`；注册入口 `t10_hidden_mass_balance_beam_a` / `t10_hidden_mass_balance_beam_b`。
- A 入口共同提供原 PhysCaP 式交互门控称重，B 入口只提供运动/横梁姿态；B 不将去掉测量的 PhysCaP 冒称原方法。
- 新变体固定支座并沿世界 Y 放置，初始物块在 x=0.26 m 一侧；原 Stock 的 freejoint 与合法中心放置保持原定义。横梁仍由真实 hinge、弹簧、阻尼、摩擦及接触模拟。
- 本版公开弹簧刚度为0.80 N m/rad、hinge frictionloss为0.001 N m，支座高度0.085 m。固定支座与横梁显式排除同组件自碰撞，保留真实hinge及物块/梁/桌面接触。开发回合已走通名义装载与原15cm称重宏；首轮baseline验证放置、反馈和未见载荷表现。
- 允许合法叠放，支撑接触链连接横梁且没有桌面/机器人额外承重；最终区域、姿态与释放保持窗按公开说明评分。是否有 `if` 不参与任务评分。

接入层与任务主体由 Engineer 实现，main完成最后源码修正和构建验收。构建记录：`outputs/acceptance_validation/hidden_mass_balance_build.log`。7个Python文件、A/B draft与oracle源码均编译通过，场景模型编译为17 body / 19 geom / 30 qpos；未创建episode或推进仿真。

`build_hidden_mass_balance_model(theta)`构建MuJoCo模型；`masses_by_block(theta)`把连续latent映射到物块；`nonstack_feasibility_witness(theta)`提供分布可行性的一个二维摆法。见证不参与任务成功评分。TaskBackend提供盲装载draft和特权oracle源码，尚未验证它们的真实抓放表现。

本次注册拒绝legacy `physcap`及T10不适用的`physcap_primitives`消融；正式`physcap_appendix`仅参加A轨。固定柔顺/称重解析/倾角反馈在新任务上的专用控制实现和性能实验属于后续阶段。

当前公共场景几何与载荷分布参数集中在上述两个模块；没有新增配置框架。独立KinDER环境可承载官方任务，现有RobotAPI场景适配目前覆盖这个平衡装载任务，后续任务按原生对象及接入模式新增。

## 后续快速新增任务

1. 从 `third_party/kindergarden/src/kinder/envs/dynamic3d/tasks/` 的具体 JSON 和对象类选模板，记录来源与本地变更。
2. 复用对象/资产拼接与现有 MujocoEpisode，只在新的 task 模块实现参数分布、公开任务说明、评分和注册。
3. 观测只暴露任务契约允许的量。额外测量在开发/测试/online/twin 路径使用同一个实现和成本，不能只给最终评估或单个方法。
4. 本地任务的控制/对象尺度变更与官方任务定义分开记录。新增接口只服务当次任务，不预建通用配置框架。

本轮按 lean-performance-coding 限定范围实现。未经用户明确要求，不新增/修改/运行测试、示例、smoke 或 benchmark；仅可读源码、检查差异，以及进行不推进物理的编译/构建。安装成功不等于抓放、测量或任务性能已验证。
