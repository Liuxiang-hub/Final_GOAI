# Final GOAI · Dual-PIPER VLA

> 面向 GOAI 2026 决赛双 PIPER X 六任务的视觉—语言—动作模型训练与部署工程。

本项目以 **LingBot-VLA 2.0** 为主体模型，使用官方真实机器人演示数据完成 LeRobot v3 数据整理、机器人特征映射、训练集归一化、动作专家微调以及后续真机部署准备。

当前目标很明确：让模型在保持通用视觉语言能力的同时，学会双 PIPER X 的协同动作、夹爪控制、空间关系与未来状态变化。

---

## 项目概览

| 项目 | 当前方案 |
|---|---|
| 机器人 | 双 PIPER X，12 维机械臂关节 + 2 维夹爪 |
| 任务数量 | 6 个真实机器人任务 |
| 演示数据 | 600 episodes |
| 数据格式 | LeRobot v3.0，25 FPS |
| 训练 / 验证 / 测试 | 510 / 60 / 30 episodes |
| 主模型 | LingBot-VLA 2.0 6B |
| 视觉语言骨干 | Qwen3-VL-4B-Instruct |
| 动作建模 | 36 层稀疏动作 MoE，32 experts，Top-4 |
| 动作窗口 | 50 steps |
| 空间监督 | MoGe + MoRGBD |
| 时序监督 | DINO Video |
| 当前训练策略 | 冻结 Qwen3-VL，训练动作 MoE、投影层与对齐头 |

## 为什么采用 Expert-only

训练集只有 510 条轨迹，直接进行 6B 全参数训练容易破坏预训练视觉语义，并增加过拟合和显存压力。

第一阶段采用更稳健的参数范围：

- 冻结 Qwen3-VL 视觉语言主体；
- 冻结 MoGe、MoRGBD 和 DINO Video 教师；
- 训练动作 MoE；
- 训练状态与动作投影层；
- 训练当前/未来深度与视频特征对齐头。

静态统计约为：

| 模块 | 参数量 | 状态 |
|---|---:|---|
| Qwen3-VL | 4.438B | 冻结 |
| 动作 MoE | 1.787B | 训练 |
| 深度/视频对齐模块 | 约 123M | 训练 |
| 其他动作侧模块 | 约 28M | 训练 |
| 总参数 | 6.376B | 约 1.938B 可训练 |

## 模型中的两类教师

教师模型只生成监督目标，不参与参数更新：

```text
三路相机图像
      │
      ├── MoGe + MoRGBD ──> 当前/未来深度与空间结构监督
      │
      └── DINO Video ─────> 当前到未来的视觉状态变化监督
                                  │
Qwen3-VL 特征 ──> 可学习对齐头 ────┤
                                  ▼
                              动作 MoE
                                  │
                                  ▼
                       双臂关节 + 双夹爪动作
```

对齐头承担“特征翻译”工作：把 VLA 内部特征转换到教师特征空间，使动作模块不仅拟合关节数值，也学习物体远近、相对位置和未来变化。

## 数据约定

转换后的数据包含：

- 600 episodes；
- 666,002 frames；
- 3 路相机：顶部、左腕、右腕；
- 原始状态/动作 14 维；
- 统一映射为 12 维双臂关节和 2 维夹爪；
- 动作采用 `joint_delta` 相对轨迹；
- 每个训练样本预测未来 50 步动作。

数据划分使用固定种子，并保证三个集合无重叠：

| Split | Episodes | 每任务 |
|---|---:|---:|
| Train | 510 | 85 |
| Validation | 60 | 10 |
| Test | 30 | 5 |

归一化统计只允许使用训练集计算，验证集和测试集不得参与。

> 数据集、模型权重和训练检查点体积较大，不存放在 Git 仓库中。

## 推荐训练配置

RTX 6000D 84GB 单卡建议从以下配置开始：

```yaml
train:
  data_parallel_mode: ddp
  vlm_fsdp: false
  module_fsdp_enable: false
  enable_full_shard: false

  freeze_vision_encoder: true
  train_expert_only: true
  train_state_proj: true

  enable_gradient_checkpointing: true
  enable_mixed_precision: true
  enable_fp32: false
  use_compile: true

  optimizer: muon
  lr: 1.0e-5
  lr_warmup_ratio: 0.05

  micro_batch_size: 2
  gradient_accumulation_steps: 4
  global_batch_size: 8
```

正式训练前必须完成单步 `forward + backward + optimizer.step` 压力测试，再根据峰值显存决定是否把 micro batch 提升到 4。

## Step 与 Epoch

训练集展开后共有 568,610 个逐帧动作窗口。设全局批量为 `G`：

```text
steps_per_epoch = floor(568610 / G)
```

当 `global_batch_size = 8` 时：

```text
1 epoch = 71,076 steps
10,000 steps ≈ 0.141 epoch
40,000 steps ≈ 0.563 epoch
60,000 steps ≈ 0.844 epoch
```

因此本项目按 step 和验证结果选择模型，不把“完整跑若干 epoch”作为默认目标。

## 训练阶段

1. 环境与 CUDA/FlashAttention 验证；
2. 权重、数据、Robot Config 与归一化统计检查；
3. 单步反向传播显存测试；
4. 20～50 steps 冒烟训练；
5. 1,000 steps 阶段检查；
6. 5,000 / 10,000 steps 候选检查点；
7. 六任务分项离线验证；
8. 云真机与 PIPER X 实机 A/B 测试。

训练不会默认一次跑满。每个阶段必须先确认 loss、梯度、MoE 路由、显存和六任务均衡性。

## 仓库边界

本仓库计划保存：

- LingBot-VLA v2 必要代码补丁；
- 双 PIPER X Robot Config；
- 六任务训练列表与 split 清单；
- 仅训练集归一化统计；
- RTX 6000D 环境安装与启动脚本；
- 训练、验证和部署说明。
