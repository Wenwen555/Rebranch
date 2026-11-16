# ReBranch

研究通过跨实例失败与冲突证据，帮助 LLM 合成能主动探测、根据物理反馈调整动作的机器人程序。当前使用 PiPER-X RobotAPI 和 MuJoCo，任务底座复用 KinDER 的原生对象与资产。

## 当前进度

- 已实现未知载荷平衡装载任务：三个同外观物块、连续隐藏质量、公开左右装载区域。允许连续落点、重抓和合法叠放；释放并撤离后，横梁须在 0.5 秒内保持倾角不超过 5°。
- A 轨向所有方法提供相同的 PhysCaP 式交互门控称重；B 轨使用物块运动和横梁姿态。任务评分检查完整足迹、真实承重与释放撤离。
- KinDER 独立环境及固定源码版本已接入。真实执行已确认名义可行装载和三个起点的原 15 cm 称重宏；放置闭环的精度正在验收。
- 已接入 PhysCaP 附录流程、AgenticGenPlan 及自定义控制/参数优化对照。任务有效性、原方法忠实度和实验表现分别验收。

2026-11-17：首轮四个无 LLM baseline 批次已启动过，当前进程已停止，完整结果尚未生成。后续先处理批次执行与放置精度，再运行固定控制、纯参数优化、称重解析和倾角辨识；PhysCaP 与 AgenticGenPlan 随后使用同一任务协议评估。

当前阶段建立 baseline 和任务证据，ReBranch 的方法优势由后续同预算比较和消融检验。

## Baseline

| 入口 | 设置 | 实现 |
|---|---|---|
| `t10_fixed_blind` | A/B | 固定程序与落点，无质量测量 |
| `t10_parameter_only` | A/B | 固定结构，仅优化左右分配与连续落点；复用 Retuner |
| `t10_mass_balance` | A | 实际称重后解析配重 |
| `t10_tilt_feedback` | A/B | 根据公开横梁姿态交互辨识、调整装载 |
| `physcap_appendix` | A | 原论文附录的仿真测量门控及规划、探索、执行流程，适配本地 RobotAPI |
| `agentic_genplan` | A/B | 自主编写诊断、实验、分析与策略，提交冻结程序后评估 |
| `oracle_twin` | A/B | 使用真实参数的特权可行性参照 |

PhysCaP 使用其仿真称重协议。AgenticGenPlan 使用本地 RobotAPI 适配。具体来源与第三方许可证见 [PhysCaP 来源说明](rebranch/baselines/physcap_appendix_assets/SOURCE.md)及 [KinDER 接入说明](KINDER.md)。

## 安装与运行

使用 Python 3.12、`uv` 和固定的 KinDER Dynamic3D 依赖：

```bash
git submodule update --init third_party/kindergarden
git -C third_party/kindergarden apply ../../requirements/kinder-egl.patch
uv venv --python 3.12 .venv-kinder
uv pip install --python .venv-kinder/bin/python -r requirements/kinder-dynamic3d.lock
uv pip install --python .venv-kinder/bin/python --no-deps -e third_party/kindergarden -e .

export MUJOCO_GL=egl DISABLE_AUTO_DYNAMIC3D_SCENES_DOWNLOAD=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

任务入口为 `t10_hidden_mass_balance_beam_a` 和 `t10_hidden_mass_balance_beam_b`。无 LLM 对照可使用 `mock` provider，该方法本身不调用模型：

```bash
.venv-kinder/bin/python scripts/run_method.py \
  --task t10_hidden_mass_balance_beam_a --method t10_mass_balance \
  --provider mock --workers 1 --n-hidden 50 --n-ood 20 \
  --output-dir outputs/t10_mass_balance
```

LLM baseline 使用相同入口及 `--provider codex --model gpt-6.1-sol --effort low`，并显式指定开发与在线请求预算。结果保存在独立输出目录，开发、验证和最终测试分开，报告成功率、失败类型、物理交互与模型成本。

## Roadmap

- [ ] 完成 T10 放置、称重和反馈控制的真实执行验收，冻结任务与评价协议。
- [ ] 跑完固定控制、纯参数优化、称重解析、倾角反馈、PhysCaP 和 AgenticGenPlan 的配对 baseline。
- [ ] 统一开发实例访问权限和 rollout/token 预算，比较普通修复与显式冲突证据组织。
- [ ] 用多随机种子、未见载荷/布局和探测读数干预，检验合成可靠性、资源效率及探测的实际作用。

## 代码结构

`rebranch/tasks` 定义任务，`rebranch/sim` 提供物理与 KinDER 接入，`rebranch/baselines` 实现对照，`rebranch/eval` 与 `scripts` 执行实验。`third_party/kindergarden` 固定为上游 commit `5b2dbac482142b879b48449b3fa2af63f641aed3`。

仓库保留原有 `active_sysid`、`preview_tools` 和 `agentic_rsr` 原型；当前 ReBranch 实验入口为 `scripts/run_method.py`。
