# 🛠️ GOAI 2026 双 PIPER 复现指南

本仓库只保存团队原创配置、数据处理脚本、固定划分和 LingBot-VLA 2.0 适配补丁。原始数据与模型权重请从各上游官方仓库下载。

## 1. 📥 准备上游代码

```bash
git clone https://github.com/Robbyant/lingbot-vla-v2.git
cd lingbot-vla-v2
git apply ../Final_GOAI/patches/lingbot-vla-v2/episode_split_loader.patch
```

随后把以下文件复制到 LingBot-VLA 对应位置：

```text
Final_GOAI/configs/goai_piper_x.yaml
  -> lingbot-vla-v2/configs/robot_configs/goai_piper_x.yaml

Final_GOAI/configs/train_expert_only.yaml
  -> lingbot-vla-v2/configs/vla/real_robot/goai_piper_x_expert_only.yaml

Final_GOAI/assets/norm_stats/goai_piper_x.json
  -> lingbot-vla-v2/assets/norm_stats/goai_piper_x.json

Final_GOAI/assets/training_data/goai_piper_x_six_tasks.example.txt
  -> lingbot-vla-v2/assets/training_data/goai_piper_x_six_tasks.txt
```

将配置文件和训练列表中的 `/path/to/...` 改成实际数据、权重和输出目录。

## 2. 🔄 转换数据

```bash
python scripts/data/convert_real_hdf5_to_lerobot_v30_joint.py \
  --source /path/to/GOAI-2026/data/real \
  --output /path/to/data/lerobot_v30_joint \
  --decode-workers 8 \
  --decode-batch-size 32
```

转换器保留三路视频和 14 维双臂状态，并将下一帧状态作为动作目标。

## 3. 🔍 验证与划分

```bash
python scripts/data/validate_lerobot_v30_joint.py \
  --root /path/to/data/lerobot_v30_joint \
  --expected-episodes 600

python scripts/data/create_lerobot_episode_splits.py \
  --dataset-root /path/to/data/lerobot_v30_joint \
  --seed 2026 \
  --train-per-task 85 \
  --val-per-task 10 \
  --test-per-task 5
```

仓库中的 `splits/` 是本项目采用的固定结果。训练加载器必须指向 `train_episodes.txt`；归一化统计只允许由这 510 个训练 episodes 计算。

## 4. 🚀 训练

完成 CUDA、FlashAttention、权重路径和单步反向传播测试后：

```bash
cd lingbot-vla-v2
python tasks/vla/train_lingbotvla.py --config configs/vla/real_robot/goai_piper_x_expert_only.yaml
```

实际训练入口请以上游当前版本为准。先运行 20–50 steps 冒烟测试，再按配置完成 8,884 steps；检查点保存于 2,221 / 3,332 / 4,442 / 5,553 / 6,663 / 7,774 / 8,884。

## 5. 🤖 部署当前离线候选

当前离线选择为 `global_step_7774`（1.75 epoch），选择依据与完整指标见 `configs/selected_model.yaml` 和 `assets/evaluation/full_episode_checkpoint_comparison.json`。服务端仍返回完整 50-step chunk；客户端执行 15 步后重观测，并按 `configs/deploy_temporal_adaptive.yaml` 使用四块时序集成、共识门控、自适应 EMA 和振荡抑制。

```bash
cd /path/to/lingbot-vla-v2
export MODEL_PATH=/path/to/global_step_7774/hf_ckpt
export EXECUTION_HORIZON=15
bash /path/to/Final_GOAI/scripts/deploy/start_lingbot_vla_v2_server.sh
```

在连接机械臂前，必须依次完成输出维度、反归一化、关节顺序/单位/方向、夹爪范围、限位、速度/加速度、通信超时和急停验证。先空载低速运行，再逐任务闭环测试。离线最优不等于真机成功率最优。

## 6. 🔐 安全与许可证

- 不要将 SSH 密钥、密码、访问令牌或服务器地址提交到 Git。
- 不要提交原始数据、模型权重、检查点、缓存和训练日志。
- 使用者需分别遵守 GOAI 数据集、LingBot-VLA、Qwen3-VL、MoGe、DINO 与 PIPER SDK 的许可证。

