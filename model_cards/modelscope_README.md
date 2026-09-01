# GOAI 2026 多任务双臂 LingBot-VLA 2.0

> 华中科技大学人形机器人团队面向 GOAI 2026 全球赛开发的双 PIPER X 视觉—语言—动作模型。

本仓库发布当前离线最优模型 **LingBot-VLA 2.0 `global_step_8884`（2.00 standard epochs）**。仓库名 `FinalGOAI_8848` 刻意采用“8848 钛金手机”梗，作为项目的趣味命名；模型权重、评估记录和部署配置对应的真实训练步数为 **8,884**，两者不要混淆。

## 模型概览

| 项目 | 内容 |
|---|---|
| 主体模型 | LingBot-VLA 2.0 |
| 机器人 | 双 PIPER X，双臂 12 关节 + 双夹爪，共 14 维 |
| 输入 | 语言指令、顶部相机、左右腕部相机、当前机器人状态 |
| 输出 | 完整 50-step action chunk |
| 后训练方式 | Expert-only，仅训练动作 MoE、状态/动作投影与对齐头 |
| 训练数据 | GOAI 六任务 510 条训练 episodes |
| 归一化 | 仅使用 510 条训练 episodes 统计 |
| 最终 checkpoint | `global_step_8884` |
| 真机状态 | 离线验证完成，双 PIPER 低速闭环验证待完成 |

## 六项任务

1. Pen Refill Replacement
2. Plug Charger into Socket
3. Place Objects into Basket
4. Stack Blocks
5. Arrange Bowls
6. Place Bottles

## 训练与模型选择

- 训练集：510 episodes
- 验证集：60 episodes，每任务 10 条
- 测试集：30 episodes，每任务 5 条
- 全局批量：128
- 优化器：Muon
- 峰值学习率：`1e-5`
- Warmup：5%
- 正式训练：8,884 optimizer steps
- 候选比较：1.50 / 1.75 / 2.00 epochs

最终模型只由 60 条完整验证 episodes 选择。冻结模型和后处理参数后，再对 30 条完整测试 episodes 做一次全长度确认。

| 数据集 | Episodes | Frames | MSE ↓ | MAE ↓ | Velocity RMSE ↓ | Stationary RMS ↓ | Jerk RMS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 60 | 64,811 | 0.003849 | 0.028888 | 0.016001 | 0.011165 | 0.035607 |
| Frozen Test | 30 | 32,581 | 0.003722 | 0.028960 | 0.015787 | 0.011050 | 0.035157 |

> 注意：30 条 test episodes 在流程纠正前曾被用于前 500 帧快速初筛，因此不能宣称为从未观察过的严格独立测试。当前结果是冻结后的完整序列确认，离线误差也不等于真机任务成功率。

## 权重文件说明

这是一个 checkpoint，由三个 safetensors 分片共同组成，并不是三个独立模型：

```text
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
```

加载器通过索引自动组合三个分片。Tokenizer、预处理器和模型配置文件必须与权重一起保留。

## 闭环部署与动作消抖

模型服务端始终返回完整 50 步预测。机器人客户端执行 15 步后重新观测并再次推理，不能把 50 步全部开环执行。

当前部署后处理位于 `deployment/`：

```text
deployment/
├── deploy_temporal_adaptive.yaml
├── action_chunk_blender.py
└── temporal_ensemble_filter.py
```

正式离线配置包含：

- 15-step 闭环重规划；
- 最近 4 个 action chunks 按绝对控制时刻对齐；
- 共识门控抑制相互矛盾的预测；
- 自适应 EMA，在快速动作和静止保持之间切换；
- 7-step 振荡检测，识别连续小幅反向摆动；
- 累积死区关闭，避免产生“保持—跳变”台阶。

消抖属于客户端部署逻辑，不在 safetensors 权重内部。只下载模型权重可以得到原始 50-step 动作，但不会自动获得上述时序融合效果。

## 下载

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    "LiuXiangg/FinalGOAI_8848",
    revision="master",
)
print(model_dir)
```

## 使用前检查

连接真实机械臂前必须确认：

1. 三路相机名称、顺序、分辨率与训练配置一致；
2. 14 维关节/夹爪顺序、单位、零点和正负方向一致；
3. 状态归一化与动作反归一化使用训练集统计；
4. `joint_delta` 正确转换为控制指令；
5. 设置关节限位、速度/加速度限制、通信超时和急停；
6. 先空载、低速、短窗口测试，再进入六任务闭环验证。

## 局限性

- 当前只完成离线开环回放与全 episode 误差评估；
- 尚未给出双 PIPER 真机成功率；
- 后处理阈值仍需在真实控制频率和动作坐标系下复核；
- 不应仅凭最低 loss 判断比赛最终模型。

## 工程仓库

完整训练配置、固定数据划分、评估图和部署代码：

- GitHub: https://github.com/Liuxiang-hub/Final_GOAI
- LingBot-VLA 2.0: https://github.com/Robbyant/lingbot-vla-v2

## 许可证与数据合规

使用者需分别遵守 GOAI 数据集、LingBot-VLA、Qwen3-VL、MoGe、DINO 和 PIPER SDK 的许可证。模型发布不代表重新授权原始比赛数据；未经许可不要公开上传原始真机数据。
