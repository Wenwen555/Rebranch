# ReBranch

项目仓库：<https://github.com/Wenwen555/Rebranch>

ReBranch 研究如何利用跨实例失败与冲突证据，辅助大语言模型合成能够主动探测（active probe）并依据物理反馈调整动作的机器人程序，重点考察自动程序合成的可靠性与资源效率。交互称重、解析配重与姿态反馈控制本身已有成熟解法，本项目关注模型在受限交互预算下自动合成此类控制程序的机制。现阶段工作聚焦于物理任务设计、对照方法实现与底层执行证据验收，ReBranch 合成算法优化暂时挂起。

## 任务设计与双轨接口

实验环境基于 PiPER-X RobotAPI 与 MuJoCo 构建，接入 KinDER（上游 `kindergarden` 固定于提交 `5b2dbac482142b879b48449b3fa2af63f641aed3`）的 `GeneratedSeesaw` 与 `Cube` 原生 MJCF 及相关资产，构成受 KinDER 启发的 PiPER-X 平台变体任务（详见 [KinDER接入](KINDER.md)）。

当前任务 T10 包含三个同外观、同尺寸的立方体，隐藏质量服从连续相关分布并经随机置换；场景设固定支座真实铰链横梁，公开左右二维装载区域与横梁世界姿态。任务目标是将三个物块装上横梁、松开夹爪、机械臂撤离至少 `0.20 m`，并在 `0.50 s` 保持窗内使横梁世界倾角不超过 `5°`。判定要求物块完整足迹位于标定区域内，且物块通过真实接触链由横梁承重，排除桌面、机械臂或地面额外承重。策略允许连续落点选择、重抓、换边与合法叠放，评测不检查 Python 代码是否包含 `if` 分支。

T10 设两个任务轨道：

- `t10_hidden_mass_balance_beam_a`（A 轨）：所有方法共享 PhysCaP 式仿真交互门控称重接口，完成原版 `15 cm` 抬升与保持动作后才返回物块质量，探测动作时间计入仿真账本（详见 [PhysCaP来源](rebranch/baselines/physcap_appendix_assets/SOURCE.md)）。
- `t10_hidden_mass_balance_beam_b`（B 轨）：无质量读取接口，策略仅利用物块运动响应与横梁姿态变化完成装载。

## 方法入口

当前统一实验入口为 `scripts/run_method.py`，支持的 `--method` 标识如下：

| 方法入口 (`--method`) | 适用轨道 | 机制说明 |
| :--- | :--- | :--- |
| `t10_fixed_blind` | A / B | 固定程序落点控制，无质量测量 |
| `t10_parameter_only` | A / B | 固定程序结构，通过 Retuner 搜索左右侧选择与落点参数 |
| `t10_mass_balance` | A | 执行门控交互称重后枚举分配并解析计算配重落点 |
| `t10_tilt_feedback` | A / B | 基于横梁姿态反馈进行交互辨识与装载调整 |
| `physcap_appendix` | 仅 A | 本地 RobotAPI 适配原论文附录的仿真测量门控及规划、探索、执行流程 |
| `agentic_genplan` | A / B | 本地 RobotAPI 适配自主编写诊断、实验、分析与策略并提交冻结程序 |
| `oracle_twin` | 特权参照（单列） | 读取特权真实物理参数的可行性参照 |

## 当前进度

开发验证中，名义可行装载已跑通完整成功回合，原版 `15 cm` 质量测量宏在三个初始起点均可正常揭示质量；放置控制精度与真实成功率仍在继续验收。

四个无 LLM 对照（A 轨 `t10_fixed_blind`、A 轨 `t10_parameter_only`、A 轨 `t10_mass_balance`、B 轨 `t10_tilt_feedback`）正在后台执行，统一使用 `seed=0`、共同评测集 `hidden=50` 与 `boundary=20`，开发集与最终测试集分离，纯参数优化的开发 rollout 上限设为 `2000`。目前固定盲控已跑完 `24` 个开发回合，纯参数优化正在运行，完整最终测试成绩尚未产出。`physcap_appendix` 与 `agentic_genplan` 已按原论文与附录机制完成本地 RobotAPI 适配，尚未运行 T10 真实模型实验。

## 代码结构与运行方法

仓库核心目录包括 `rebranch/tasks`、`rebranch/sim`、`rebranch/baselines`、`rebranch/eval` 与 `scripts`，同时保留早期的 `active_sysid`、`preview_tools` 与 `agentic_rsr` 原型实现。

环境安装与初始化：

```bash
git clone git@github.com:Wenwen555/Rebranch.git
cd Rebranch
git submodule update --init third_party/kindergarden
git -C third_party/kindergarden apply ../../requirements/kinder-egl.patch
uv venv --python 3.12 .venv-kinder
uv pip install --python .venv-kinder/bin/python -r requirements/kinder-dynamic3d.lock
uv pip install --python .venv-kinder/bin/python --no-deps -e third_party/kindergarden -e .
export MUJOCO_GL=egl DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

单方法运行示例（固定手写与数值对照使用 `--provider mock`，不调用大语言模型）：

```bash
.venv-kinder/bin/python scripts/run_method.py \
  --task t10_hidden_mass_balance_beam_a --method t10_mass_balance \
  --provider mock --workers 1 --n-hidden 50 --n-ood 20 \
  --output-dir outputs/t10_mass_balance
```

运行大语言模型基线时使用 `--provider codex --model gpt-6.1-sol --effort low`，并显式传入 `--max-rollouts`、`--max-llm-calls` 与 `--runtime-llm-budget` 控制预算。

当前 T10 四组无 LLM 对照批量运行命令：

```bash
.venv-kinder/bin/python scripts/run_t10_baseline_pilot.py \
  --seed 0 --max-rollouts 2000 --n-hidden 50 --n-boundary 20 \
  --workers 4 --output-dir outputs/t10_baseline_pilot
```

各次运行结果写入独立的 `outputs/` 目录，统计任务成功率、任务失败与执行错误分类，以及物理交互回合与模型调用成本。

## Roadmap

1. 完成放置、交互称重与姿态反馈控制的执行验收，冻结评测协议。
2. 产出四个无 LLM 对照及 `physcap_appendix`、`agentic_genplan` 的配对实验结果。
3. 统一开发实例访问权限及 rollout 与 token 预算，对比常规程序修复与显式跨实例冲突组织。
4. 在多随机种子、未见载荷与布局、探测读数干预条件下，检验合成可靠性、资源效率与探测信息的实际作用。
