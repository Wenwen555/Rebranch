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

## 实验结果

### 公共终态反馈先导试验（`t10-terminal-feedback-v1`）

在物理协议 `t10-v2-noslip2` 与 `seed 0` 下完成 4-job 配对评测（每 job 上限 8 模型请求、2000 dev rollouts）。所有程序先冻结再于新 `theta` 抽样集（`hidden` `2026111701` 50 例 + 域内边缘偏置 `boundary` `2026111702` 20 例）运行，共 280 计分与 4 个 nominal，最终评测 0 执行错误、0 在线 LLM；四 job 合计 19 模型请求、511667 tokens、1556 dev rollouts。

| 方法 | 轨道 | Hidden | Boundary | 开发物理回合 / 模型请求 |
| --- | --- | --- | --- | --- |
| `ReBranch` | A | 50/50 | 20/20 | 500 / 1 |
| `AgenticGenPlan` | A | 50/50 | 20/20 | 48 / 7 |
| `ReBranch` | B | 50/50 | 20/20 | 1001 / 3 |
| `AgenticGenPlan` | B | 0/50 | 0/20 | 7 / 8 |

A 轨成功率持平，ReBranch 模型请求更少，AGP 物理开发与执行成本更低；两项 ReBranch 均通过普通 `REPAIR` 生成 probe，未触发专门 `BRANCH` 合成。AGP B 最终重写版本在第 8 次请求冻结且未作开发执行，70 例均在 probe 抓取或回收未满足自检时提前返回而未进入配重，属当前 `seed 0` 候选验证与执行差异，不表示 AGP 不能解 T10。

- **`oracle`（特权参照，非 LLM）**：已知验收集 A/B 各 70/70（共 140/140）。

<details><summary>历史评测（新增公共终态反馈前）</summary>

`seed 0` 已知历史集（旧 `theta` seed `2026`/`3026`），物理协议同为 `t10-v2-noslip2`，生成反馈协议不同（保留旧 B 70/70）：

| 方法 / 参照 | 轨道 | Hidden | Boundary | 预算与说明 |
| --- | --- | --- | --- | --- |
| `t10_fixed_blind` | A | 5/50 | 4/20 | 0 LLM（固定落点控制） |
| `t10_parameter_only` | A | 11/50 | 9/20 | 0 LLM，1992 dev |
| `t10_mass_balance`（手写称重+配重） | A | 50/50 | 20/20 | 0 LLM |
| `t10_tilt_feedback`（手写倾角反馈） | B | 50/50 | 20/20 | 0 LLM |
| `AgenticGenPlan` | A | 29/50 | 16/20 | 8 请求，4 dev |
| `AgenticGenPlan` | B | 50/50 | 20/20 | 8 请求，4 dev |

</details>

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

1. [x] 收齐 `t10-terminal-feedback-v1` 有限先导结果，核查 LLM 与 probe 数据流及真实资源效率。
2. [ ] 依据实际实验结果决定后续多 seed 或观测干预实验。
3. [ ] 在具备明确机制证据的任务上实现 `SandboxDiagnosticReport`（配对噪声冲突、完整 probe 特征与分布阈值标定，支持连续反馈，仅在有证据时启用分割）。
