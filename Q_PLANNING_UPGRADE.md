# Q-Planning 升级方案

## 定位

当前正式初始方案保持不变：冻结的 LingBot-VLA 2.0 `global_step_8884` 输出
50-step、14维双 PIPER 动作块，客户端执行15步后重规划，并使用四块时序集成、
自适应 EMA、振荡抑制和机器人安全限制。

升级方案在 LingBot 与现有后处理之间增加一个 Q-Planning 候选决策层。它不会替换
或继续训练 `global_step_8884`：LingBot 批量生成 N 个候选动作块，Q-function 对候选
评分，然后使用 Q-softmax 权重融合高价值候选。参考论文与代码：

- [Q-Planning paper](https://arxiv.org/abs/2608.21204)
- [Project page](https://q-planning.github.io/)
- [Official code](https://github.com/varungiridhar/qplanning-code)

## 两条路径必须隔离

| 路径 | 状态 | 用途 |
|---|---|---|
| Baseline：step8884 + 已冻结后处理 | 当前默认 | 第一轮低速真机闭环与可靠回退 |
| Upgrade：step8884 + Q-Planning + 同一后处理 | 默认关闭 | 收集基线结果后进行 A/B 验证 |

配置 `configs/qplanning/offline_prepare.yaml` 中 `enabled: false` 是安全门。未通过
候选重放、延迟、安全回退和低速真机测试前，不得改为正式入口。

## 已经完成的前置准备

1. 固定候选动作契约为 `[N, 50, 14]`，动作语义与训练完全一致；
2. 固定 `global_step_8884` 为冻结 BC，归一化统计仍为训练集统计；
3. 提供数值稳定的 Q-softmax 加权和 top-k elite 支持；
4. 提供 LingBot 候选采样适配接口，禁止另建相机预处理路径；
5. 定义成功/失败 episode manifest 及校验工具；
6. 固定接入顺序与 baseline fallback；
7. 提供 CPU 单元测试与静态 preflight。

运行：

```bash
pip install -r requirements-qplanning.txt
python scripts/qplanning/preflight.py
python -m unittest tests.test_qplanning_preparation
```

官方 `qplanning-code` 应安装在单独环境或单独 checkout 中；不要让其依赖覆盖已经验证的
LingBot-VLA 2.0 环境。两者通过 adapter/API 边界连接。

## 真机失败数据之前还能做什么

### A. LingBot 多候选采样适配

在 GPU 服务器复用已经验证过的 LingBot 图像、语言、状态预处理和模型实例，实现
`sample_action_chunks(..., n_samples=N)`。必须只编码一次观测，再批量采样，避免耗时
随 N 完整倍增。先验证 N=1 的输出与当前基线完全一致，再测试 N=4。

### B. 成功演示上的 Q 离线初始化

官方方法允许用同一批成功 demonstrations 初始化 Q。这里只能让 Q 学习“朝终点推进
时价值通常上升”，无法充分学习失败边界。因此它是预训练，不是升级有效性的证据。
训练仍只能使用510条训练 episodes；60条 validation 不进入 replay buffer，30条 test
保持冻结。

### C. 候选重放与延迟基准

对固定 validation observation 保存 N 个候选、Q 分数、融合动作和 baseline 动作，检查：

- 归一化只执行一次，14维顺序和单位不变；
- Q 融合动作没有 NaN、越界或明显离开候选包络；
- N=4 时规划 p95 小于500ms，为15步/25Hz形成的600ms预算留出安全余量；
- Q 异常、超时或模型缺失时无状态切回 baseline 第一候选；
- Q 选择后继续使用相同的时序集成与安全限制。

## 真机阶段需要记录的数据

每个完整 episode 保存三路 RGB、14维状态、所有候选动作、候选 Q 值、融合动作、
最终执行动作、任务、终止原因、耗时和 episode 级 `success: true/false`。manifest 每行：

```json
{"episode_id":"baseline-fill-0001","task":"fill_pen_holder","success":false,"frames":250,"source":"baseline_rollout","data_path":"episodes/baseline-fill-0001"}
```

成功标签可以在 episode 结束后人工给出；人不需要提供纠正动作。安全急停、通信超时和
人工中止必须保留为独立终止原因，不能伪装成普通失败。

## 上线门槛

1. baseline 先完成六任务低速真机测试，并能在至少部分任务产生成功轨迹；
2. N=1 一致性、N=4延迟、归一化和 fallback 测试全部通过；
3. 离线 held-out replay 上，Q 排序优于随机且不增加越界率；
4. 只选择一个中等难度任务进行低速 A/B；
5. 安全停止率不升高且成功率置信区间有改善后，才扩展到六任务；
6. 不以 Q-loss 单独决定是否上线，最终依据真机成功率、最差任务、安全停止与p95延迟。

## 尚未完成/当前阻塞

- LingBot 2.0 服务器内部的批量 Flow-Matching 采样 API；
- 与官方 Q-network 编码器和训练入口的实机环境集成；
- episode 级成功检测或人工标签界面；
- 基线真机成功/失败 rollouts；
- Q-only 自改进训练与真实 A/B 结果。

这些项目需要 GPU 服务器上的实际 LingBot 对象或真机数据，不能用离线动作预测图替代。

## RGB 数据链路审计（2026-09-05）

正式服务器上对原始 `episode_0000000` 的顶部、左腕、右腕相机分别抽取首帧、中间帧和
末帧，共9帧。历史转换函数与官方 `XPolicyLab.utils.process_data.decode_image_bit` 的
输出均逐像素一致（MAE=0）。LeRobot H.264视频首帧按RGB读取后相对原始帧MAE约1.786，
错误交换红蓝通道时约4.851；差异符合有损视频编码，颜色顺序正确。因此当前step8884
不存在由该转换造成的BGR/RGB域错配，不需要据此重训。

为满足官方接口约束并兼容其他bit布局，转换脚本已经改为直接调用
`decode_image_bit`；后续转换环境必须能导入官方XPolicyLab包，禁止恢复手写PIL或
`cv2.imdecode`路径，也禁止在解码后额外执行BGR/RGB交换。
