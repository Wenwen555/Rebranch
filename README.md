# ReBranch

项目仓库：<https://github.com/Wenwen555/Rebranch>

ReBranch 研究如何利用跨实例失败与冲突证据，辅助大语言模型合成能够主动探测（active probe）并依据物理反馈调整动作的机器人程序，重点考察自动程序合成的可靠性与资源效率。交互称重、解析配重与姿态反馈控制本身已有成熟解法，本项目关注模型在受限交互预算下自动合成此类控制程序的机制。当前阶段 1 已完成 T10 物理执行契约验证并冻结协议 `t10-v2-noslip2`，正在运行对照与基线实验；阶段 2 的 `SandboxDiagnosticReport` 与 ReBranch 合成算法优化尚未实施。

## 任务设计与双轨接口

实验环境基于 PiPER-X RobotAPI 与 MuJoCo 构建，接入 KinDER（上游 `kindergarden` 固定于提交 `5b2dbac482142b879b48449b3fa2af63f641aed3`）的 `GeneratedSeesaw` 与 `Cube` 原生 MJCF 及相关资产，构成受 KinDER 启发的 PiPER-X 平台变体任务（详见 [KinDER接入](KINDER.md)）。

当前任务 T10 包含三个同外观、同尺寸的立方体，隐藏质量服从连续相关分布并经随机置换；场景设固定支座真实铰链横梁，公开左右二维装载区域与横梁世界姿态。任务目标是在每回合 `200` 次 RobotAPI 调用上限内将三个物块装上横梁、松开夹爪、机械臂撤离至少 `0.20 m`，并在 `0.50 s` 保持窗内使横梁世界倾角不超过 `5°`。判定要求物块完整足迹位于标定区域内，且物块通过真实接触链由横梁承重，排除桌面、机械臂或地面额外承重。

针对部分装载斜面实测出现的 MuJoCo 软约束接触漂移，冻结协议 `t10-v2-noslip2` 在任务初始化时对所有 A/B 轨方法与基线统一设置 `opt.noslip_iterations=2`，摩擦系数、质量分布与成功判据均未放宽。策略允许连续落点选择、重抓、换边与合法叠放。评测元数据 `requires_branching` 仅用于指标统计标记，物理成功与评测均不要求代码包含 Python `if` 分支；最重物块类别（heaviest-class）纯度仅作执行路径诊断；B 轨因连续反馈分类尚未实现，探测率指标记为 `null`。

T10 设两个任务轨道：

- `t10_hidden_mass_balance_beam_a`（A 轨）：所有方法共享 PhysCaP 式仿真交互门控称重接口，完成原版 `15 cm` 抬升与保持动作后才返回物块质量，抓取失败返回 `NaN`，探测动作时间计入仿真账本（详见 [PhysCaP来源](rebranch/baselines/physcap_appendix_assets/SOURCE.md)）。
- `t10_hidden_mass_balance_beam_b`（B 轨）：无质量读取接口，策略仅利用物块运动响应与横梁姿态变化完成装载。

## 方法入口

当前统一实验入口为 `scripts/run_method.py`，支持的 `--method` 标识如下：

| 方法入口 (`--method`) | 适用轨道 | 机制说明 |
| :--- | :--- | :--- |
| `t10_fixed_blind` | A / B | 固定程序落点控制，无质量测量 |
| `t10_parameter_only` | A / B | 固定程序结构，通过 Retuner 搜索左右侧选择与落点参数 |
| `t10_mass_balance` | A | 执行门控交互称重后枚举分配并解析计算配重落点 |
| `t10_tilt_feedback` | A / B | 基于横梁姿态反馈进行交互辨识与装载调整 |
| `physcap_appendix` | 仅 A | 本地 RobotAPI 适配原论文附录的仿真测量门控及规划、探索、执行流程（不参加 B 轨） |
| `agentic_genplan` | A / B | 本地 RobotAPI 适配自主编写诊断、实验、分析与策略并提交冻结程序 |
| `oracle_twin` | 特权参照（单列） | 读取特权真实物理参数的可行性参照 |

平台适配边界说明：`agentic_genplan`（AGP）提示词历史保留最近 `2` 次详细检查记录并保留原始 `run_analysis` 可读，开发噪声种子采用 `seed + case_index`；AGP 与 ReBranch 共享相同的 `50000` 行 Python 执行上限；原生 AGP 环境可使用 `numpy`/`scipy`，ReBranch 执行沙箱使用标准库 `math`。

## 当前进度

阶段 1 已冻结 `t10-v2-noslip2` 协议并完成底层契约验收：参考控制采用分段微张松爪撤离以避免指垫碰撞邻块，并按公开物块四元数在横梁法向的投影计算释放高度；配合统一启用的 `noslip_iterations=2`，新协议下 `10` 项 T10 真实契约测试全部通过（含 `6` 个名义置换 oracle、桌面/夹爪额外承重拒绝、抓取失败返回质量 `NaN`），相关 `40` 项回归测试通过（未覆盖仓库全量 `pytest`）。手工强对照单回合调用数均在 `200` 上限内（A 轨 `78` 次、B 轨 `199` 次），并在新协议 `4` 个补充置换定点检查中取得 `4/4` 通过（仅作定点检查记录，不代表总体测试成功率）。协议冻结前的旧版运行结果单独留作诊断归档，不混入正式结果对比表。

新协议正式对照与基线实验正在运行，完整评测成绩尚未产出。24 个开发实例上，固定控制成功 4/24，交互称重与解析配重成功 24/24，修复后的倾角反馈控制成功 24/24（每回合 199 次调用）；这些是开发结果：

- **无 LLM 控制批次**：已启动 `t10_fixed_blind`、`t10_mass_balance`、`t10_tilt_feedback`（各 `24` 个开发回合）与 `t10_parameter_only`（开发总 rollout 上限 `<= 2000`）四组批次，统一评测 `hidden=50`、`boundary=20` 及名义检查项，并保存完整最终执行轨迹（final trace）。
- **大语言模型基线**：已统一 AGP 的工具消息格式与执行分发，相关 23 项回归测试通过；已启动 `agentic_genplan` A/B 轨与 `physcap_appendix` A 轨的顺序批次（`gpt-6.1-sol`，推理强度 `low`）。
- **倾角反馈控制**：已按公开几何修正斜梁上回收物块时的指垫净空，冻结程序通过 24 个开发实例；隐藏集与边界集另行评测。
- **阶段 2 状态**：`SandboxDiagnosticReport` 与 ReBranch 跨实例冲突合成优化尚未实施。

阶段 1 审阅与旧协议诊断归档位于 `outputs/repair_validation/stage1_review/`，新协议实验输出位于 `outputs/repair_validation/t10_v2_noslip2/`；大体量轨迹与运行产物保留在本地实验目录，未上传版本库。

## 代码结构与运行方法

仓库核心目录包括 `rebranch/tasks`、`rebranch/sim`、`rebranch/baselines`、`rebranch/eval` 与 `scripts`。

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
  --output-dir outputs/repair_validation/t10_v2_noslip2/t10_mass_balance
```

运行大语言模型基线时使用 `--provider codex --model gpt-6.1-sol --effort low`，并显式传入 `--max-rollouts`、`--max-llm-calls` 与 `--runtime-llm-budget` 控制预算。

T10 四组无 LLM 对照批量运行命令：

```bash
.venv-kinder/bin/python scripts/run_t10_baseline_pilot.py \
  --seed 0 --max-rollouts 2000 --n-hidden 50 --n-boundary 20 \
  --workers 4 --output-dir outputs/repair_validation/t10_v2_noslip2/baseline_pilot
```

各次运行结果写入独立的 `outputs/` 子目录，统计任务成功率、任务失败与执行错误分类、完整最终轨迹，以及物理交互回合与模型调用成本。

## Roadmap

1. 收齐冻结协议 `t10-v2-noslip2` 下四组无 LLM 对照及 `agentic_genplan`（A/B 轨）、`physcap_appendix`（A 轨）的完整配对评测结果。
2. 实施阶段 2 `SandboxDiagnosticReport`：从现有 `CallRecord.result` 提取公开位姿、保留早期探测记录、构建适配物理与估计噪声的跨实例冲突证据，并对条件适用的分割方法做阈值标定，兼容连续反馈控制而不强制二分类切分或 Python `if` 分支。
3. 统一开发实例访问权限及 rollout 与 token 预算，对比常规程序修复与显式跨实例冲突组织。
4. 在多随机种子、未见载荷与布局、探测读数干预条件下，检验合成可靠性、资源效率与探测信息的实际作用。
