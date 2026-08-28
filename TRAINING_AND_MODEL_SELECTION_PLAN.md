# 🧪 训练与真机前模型筛选计划

> 目标：在一天级算力窗口内完成 LingBot-VLA 2.0 第一阶段 Expert-only 后训练，并在接入双 PIPER 真机前，把候选检查点从“训练可用”逐级筛选为“安全、稳定、六任务均衡”的最终模型。

## 1. ⚡ 明日算力结论

当前正式训练目标定义为 **2 epoch = 8,884 optimizer steps**。正式使用：

> **2 × RTX 6000D 84GB，位于同一台服务器。**

双卡10-step实测在预热后约为 **10秒/optimizer step**，2 epoch纯训练约24.7小时；加上加载、数据波动和四次检查点保存，计划约26–28小时。

| RTX 6000D 数量 | 规划单步时间 | 10,000 steps 纯训练 | 加 10% 运行开销 | 一天内完成判断 |
|---:|---:|---:|---:|---|
| 1 | 14–18 s | 38.9–50.0 h | 42.8–55.0 h | 不可行 |
| 2 | 8–11.5 s | 22.2–31.9 h | 24.4–35.1 h | 临界，不保证 |
| **4** | **4.5–7.2 s** | **12.5–20.0 h** | **13.8–22.0 h** | **推荐** |
| 8 | 2.8–4.2 s | 7.8–11.7 h | 8.6–12.8 h | 可行，但性价比较低 |

这些是排期区间，不是硬件理论峰值承诺。开跑后必须用同一配置实测 100 个稳定 steps，再更新预计完成时间。

## 2. ⏱️ 时间计算原理

### 2.1 Step 与 epoch

训练集展开后共有 568,610 个动作窗口。全局批量为 `B` 时：

```text
steps_per_epoch = floor(568610 / B)
```

当前 `global_batch_size = 128`：

```text
steps_per_epoch = floor(568610 / 128) = 4,442
2 epoch = 8,884 steps
```

完整 1 epoch 若要求一天结束，平均每个 optimizer step 必须满足：

```text
86,400 / 8,884 = 9.726 s/step
```

这远低于当前模型包含三路图像、动作 MoE、深度教师和视频教师时的实际单步开销，所以明天不以完整 epoch 为目标。

### 2.2 总时间

```text
纯训练时间（小时） = max_steps × measured_step_seconds / 3600
排期时间 = 纯训练时间 × 1.10
```

额外 10% 覆盖数据抖动、首次编译、评估、日志、检查点写盘和短暂停顿。若使用更频繁的验证或保存，应提高到 15%。

### 2.3 多卡缩放

多卡不会线性加速。粗略计算为：

```text
t_N = t_1 / (N × η_N)
```

其中 `η_N` 是扩展效率，受到梯度同步、FSDP all-gather、PCIe/NVLink、数据读取和教师网络计算影响。排期可保守使用：

| GPU 数量 | 规划扩展效率 η |
|---:|---:|
| 2 | 0.75–0.85 |
| 4 | 0.60–0.75 |
| 8 | 0.45–0.60 |

此前两张 RTX 4080 SUPER 在 `torch.compile` 预热后的实测约为 **19.46 s/step**。它证明双卡链路和训练逻辑可运行，但不能直接当成 RTX 6000D 的速度；6000D 必须重新做短基准。

## 3. 🖥️ 双卡配置原理

建议保持优化目标不变：

```yaml
train:
  freeze_vision_encoder: true
  train_expert_only: true
  train_state_proj: true

  global_batch_size: 128
  micro_batch_size: 16
  gradient_accumulation_steps: 4

  enable_mixed_precision: true
  enable_fp32: false
  enable_gradient_checkpointing: true
  use_compile: false

  optimizer: muon
  lr: 1.0e-5
  lr_warmup_ratio: 0.05
  max_steps: 8884
  save_steps: 2221
```

原理：

1. **global batch = 128**：2 卡、每卡 micro batch 16、累积4次，已通过真实数据10-step测试。
2. **每卡 micro batch = 16**：峰值62.72GB/卡，预热后约10秒/optimizer step；保留约21GB物理显存余量。
3. **Expert-only**：冻结 Qwen3-VL 和教师，只更新约 1.938B 动作侧参数，降低过拟合与优化器显存。
4. **BF16/混合精度**：减少显存和 Tensor Core 计算时间；不使用全 FP32。
5. **暂不启用 `torch.compile`**：正式基线采用已经实测通过的 eager 路径，后续单独基准后再决定。
6. **同机双卡**：避免跨节点网络成为 FSDP2 同步瓶颈。
7. **DDP 与 FSDP2 实测二选一**：84GB 若能容纳 DDP 副本，DDP 通常通信更直接；若峰值显存不安全，再使用 FSDP2 full shard。选择以 100-step 基准为准。

官方4卡 A6000示例是每卡 `micro_batch_size=1`、accumulation=1、global batch=4。本项目依据双6000D 84GB实测使用global batch 128；若长期运行出现显存不稳定，回退为：

```yaml
micro_batch_size: 8
gradient_accumulation_steps: 8
global_batch_size: 128
```

无论使用哪个组合，都必须满足官方公式：

```text
global_batch_size = micro_batch_size × GPU数量 × gradient_accumulation_steps
```

## 4. 🗓️ 明日训练时间表

### 阶段 A：开跑前检查（约 45–90 分钟）

1. 核对 4 张 GPU 型号、显存、驱动、拓扑和 NCCL 通信；
2. 校验 LingBot、Qwen3-VL、MoGe/MoRGBD、DINO 权重完整；
3. 校验 510 个训练 episodes 和训练集归一化统计；
4. 跑一次 `forward + backward + optimizer.step`；
5. 分别跑 DDP 与 FSDP2 的 20-step 冒烟测试；
6. 开启 compile，丢弃预热 steps 后测量连续 100 steps。

正式配置已经通过10-step测试：无 OOM、无 NaN、双卡负载正常，稳定单步约10秒。长训前再执行100-step基准确认ETA。

### 阶段 B：正式训练

| Step | 动作 |
|---:|---|
| 0–50 | 冒烟；检查 loss、梯度、MoE 路由和显存 |
| 500 | 首次趋势检查，不用于最终选模 |
| 2,221 | 0.5 epoch：保存并运行六任务验证 |
| 4,442 | 1.0 epoch：中期评估 |
| 6,663 | 1.5 epoch：保存候选检查点 |
| 8,884 | 2.0 epoch：第一阶段终点，不自动继续全参数训练 |

正式训练每 **2,221 steps（0.5 epoch）** 保存一次，共四个候选检查点。按约43GB/检查点估算，训练前需保证至少220–250GB可用空间。

## 5. 🔎 真机前检查点筛选

### 第一级：训练健康筛选

直接淘汰出现以下情况的检查点：

- NaN/Inf、梯度爆炸或 loss 突跳；
- MoE 专家严重塌缩，少数专家长期垄断路由；
- 任一任务 loss 持续恶化；
- 推理延迟、显存或输出维度不满足部署要求；
- 关节或夹爪动作出现明显越界、抖动和不连续。

### 第二级：离线六任务筛选

对 1k–10k 的十个候选统一使用 60 条验证 episodes：

- 分任务 VLA/action loss；
- 动作 MAE/RMSE 与 action chunk 连续性；
- 末端动作误差和夹爪开闭正确率；
- 最差任务成绩，而不只看六任务平均值；
- 深度、未来深度、DINO 辅助 loss；
- MoE 路由熵与专家使用均衡度。

建议综合分数：

```text
score = 0.50 × 六任务平均分
      + 0.30 × 最差任务分
      + 0.10 × 动作平滑性
      + 0.10 × 路由与推理稳定性
```

离线阶段选出 **Top-3**，测试集不参与调参。

### 第三级：安全回放与双 PIPER 初筛

Top-3 先经过：

1. 动作反归一化检查；
2. 关节顺序、单位、方向、夹爪范围检查；
3. 速度、加速度、jerk 和关节限位检查；
4. 低速空载回放；
5. 在人工急停监护下直接进入双 PIPER，每模型 × 六任务 × 3 次，共 54 次低速初筛试验。

记录成功率、首次成功时间、人工急停次数、碰撞/越界次数和任务间最低成功率，选出 **Top-2**。

### 第四级：比赛真机最终选择

Top-2 在双 PIPER 上执行：

- 每模型 × 六任务 × 5 次，至少 60 次正式试验；
- 初始物体位置做比赛允许范围内扰动；
- 相机曝光、光照和背景做小范围变化；
- 统计总成功率与最差任务成功率；
- 若成功率接近，优先选择动作更平滑、急停更少、推理延迟更稳定的模型。

测试集只在候选方案冻结后运行一次，防止把测试集变成调参集。

## 6. 🏆 最终决策规则

1. 默认目标是 **Expert-only 2 epoch（8,884 steps）**；不启动第二阶段全参数训练。
2. 使用 **2 张 RTX 6000D 84GB**；开始100-step基准后重新计算ETA。
3. 当前实测稳定速度约10秒/step，预计总排期26–28小时。
4. 最终模型不按最低训练 loss 决定，而按“六任务平均 + 最差任务 + 安全稳定性 + 真机成功率”决定。
