# ReBranch

ReBranch（Python 包 `rebranch`，要求 Python >= 3.11，当前验证环境为 Python 3.12.15 / MuJoCo 3.3.7）研究如何利用跨实例失败与冲突证据，辅助 LLM 合成包含 active probe 并依据物理反馈调整动作的机器人控制程序，正在检验合成可靠性与交互成本。

## 当前实现与验证范围

`rebranch/core` 已实现常量提升与数值 retune、反例筛查、冲突分析、LLM 修复/分支候选生成与 admission 验证，覆盖由 LLM 参与合成且将物理探针读数用于后续动作的程序流。模块级验证方面，39 项相关测试有通过结果，T10 契约测试集单次运行 10 passed（含 6 个 oracle）。

在执行环境能力上，ReBranch 与 AgenticGenPlan（AGP）默认均限制为 50000 行 Python；AGP 可原生调用 `numpy`/`scipy`，ReBranch 仅注入 `math`。

### 候选任务：`contact_insertion`（开发中）

该模块针对把插销插入未固定插座的接触操作，公开 `peg`、`socket_box` 与 `hole` 真实位姿。作为开发中物理候选，在 SE3 修正后的 3 个开发条件中 `oracle` 与反馈控制各通过 2/3，目前尚未通过物理准入，未进入正式 LLM 比较。

## 基准任务：T10 隐藏质量平衡梁

T10 是受 KinDER 启发的 PiPER-X 平台变体，直接使用 KinDER `GeneratedSeesaw`/`Cube` 原生 MJCF 与资产（固定上游 commit `5b2dbac482142b879b48449b3fa2af63f641aed3`，接入说明见 [KINDER.md](KINDER.md)）。场景在 MuJoCo 中包含真实 hinge、弹簧与接触动力学：

- **任务目标与判据**：三个外观相同的 24 mm 物块具有隐藏的连续相关质量与随机排列；策略需将其装入横梁左右公开区域，在真实接触链由梁承重、松爪并撤离 >= 0.20 m 后的完整 0.50 s 内，横梁倾角始终 <= 5°。所有策略统一 200 API 上限，允许合法堆叠、重抓与反馈。
- **物理协议**：统一采用 `t10-v2-noslip2`（`NoSlip2`），保持质量域与成功判据不变。
- **A / B 信息轨道**（任务观测接口区分）：
  - **A 轨道**：提供共同仿真交互门控宏 `get_mass(name)`（抓取抬升 15 cm 保持后返回质量，失败返回 `NaN`，计入运动时间成本）。
  - **B 轨道**：无 `get_mass`，通过物块与横梁姿态反馈辨识质量关系。

T10 用作物理闭环与自动合成验证任务；AGP B 已解历史实例，未知质量配重本身已可由已有基线解决。

## 实验结果与在途试验

### 历史评测结果（新增公共终态反馈前，`t10-v2-noslip2`）

下表记录 `seed 0` 在已知测试集（`hidden` 50 例 + 域内边缘偏置采样 `boundary` 20 例）上的已完成生成结果。A/B 轨道各自独立生成程序；`agentic_genplan` 为 RoboCode / AgenticGenPlan 的 RobotAPI 平台适配，`physcap_appendix` 为 PhysCaP 论文附录的仅 A 轨道平台适配（参见 [SOURCE.md](rebranch/baselines/physcap_appendix_assets/SOURCE.md)）。PhysCaP A 旧在线批次已暂停，未纳入下表。

| 方法 / 参照 | 轨道 | Hidden | Boundary | 预算与执行说明 |
| --- | --- | --- | --- | --- |
| `t10_fixed_blind` | A | 5/50 | 4/20 | 0 LLM（固定落点控制，读取公开 pose 做放置反馈） |
| `t10_parameter_only` | A | 11/50 | 9/20 | 0 LLM，1992 开发 rollout |
| `AgenticGenPlan` (`agentic_genplan`) | A | 29/50 | 16/20 | 8 模型请求，4 开发 rollout |
| `AgenticGenPlan` (`agentic_genplan`) | B | 50/50 | 20/20 | 8 模型请求，4 开发 rollout；固定三次试放→读倾角→估质量→配平，测试期 0 在线 LLM，132–138 API |
| 手写称重 + 解析配重（自定义能力参照） | A | 50/50 | 20/20 | 0 LLM |
| 手写倾角反馈修后（自定义能力参照） | B | 50/50 | 20/20 | 0 LLM |

- **物理验收与特权参照（`oracle`，非 LLM 成绩）**：在已知验收集（A / B 各 50 `hidden` + 20 `boundary`）上全部通过，共 140/140，均消耗 78 API。

### 公共终态反馈协议与先导试验（`t10-terminal-feedback-v1`）

- **公平性补丁（commit `f46ce15`）**：将公共终态评分摘要由 `EpisodeResult` 经 `Trace` 同步提供给 ReBranch 与 AGP，并通过显式 allowlist 排除 `theta`、真实质量、`witness` 与质量加权 COM（公开评分反馈，不引入额外在线 sensor）。
- **四 job 先导试验**：`seed 0 × A/B × ReBranch/AGP` 已启动，待收齐结果。统一使用 `gpt-6.1-sol/low`，每 job 上限 8 模型请求、2000 开发 rollout，测试期冻结程序 0 在线 LLM；开发冻结后将在新最终集（50 + 20，共 280 计分与 4 个必要 nominal）上统一评估。

## 代码结构

- [rebranch/](rebranch)：`core`、`exec`、`llm`、`sim`、`tasks`、`baselines`、`eval`
- 工程配置与脚本：[configs/](configs)、[scripts/](scripts)、[tests/](tests)、[requirements/](requirements)
- 子模块：`third_party/kindergarden`

## 安装与运行

```bash
git clone https://github.com/Wenwen555/Rebranch.git
cd Rebranch
git submodule update --init third_party/kindergarden
git -C third_party/kindergarden apply ../../requirements/kinder-egl.patch
uv venv --python 3.12 .venv-kinder
uv pip install --python .venv-kinder/bin/python -r requirements/kinder-dynamic3d.lock
uv pip install --python .venv-kinder/bin/python --no-deps -e third_party/kindergarden -e .
export MUJOCO_GL=egl DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

统一入口为 [scripts/run_method.py](scripts/run_method.py)。以下为 0 LLM 真实物理控制运行示例（`outputs/` 大体量轨迹仅保存在本地，未上传 Git）：

```bash
.venv-kinder/bin/python scripts/run_method.py \
  --task t10_hidden_mass_balance_beam_a --method t10_mass_balance \
  --provider mock --seed 0 --workers 1 --n-hidden 50 --n-ood 20 \
  --output-dir outputs/t10_mass_balance
```

外部模型生成使用 `--provider codex --model gpt-6.1-sol --effort low` 并显式指定 `--max-rollouts` 与 `--max-llm-calls`；在线 PhysCaP 另支持 `--runtime-llm-budget` 与 `--physcap-max-turns`。

## Roadmap

1. 收齐 `t10-terminal-feedback-v1` 有限先导结果，核查 LLM 与 probe 数据流及真实资源效率。
2. 依据实际实验结果决定后续多 seed 或观测干预实验。
3. 在具备明确机制证据的任务上实现 `SandboxDiagnosticReport`（配对噪声冲突、完整 probe 特征与分布阈值标定，支持连续反馈，仅在有证据时启用分割）。
