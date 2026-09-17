# ⚡ LingBot-VLA 2.0 Real-Time Chunking

本工程将 [Real-Time Chunking](https://arxiv.org/abs/2506.07339) 接入 LingBot-VLA 2.0 的 Flow-Matching 推理链路，用于消除同步推理造成的数百毫秒停顿。RTC 不重新训练 VLA；它让机器人执行当前动作块的同时，在后台生成下一动作块，并用上一块的剩余动作约束新块。

> 当前状态：CPU 契约测试、RTX 6000D 真实 checkpoint 推理、WebSocket 闭环和延迟注入测试均已通过；双 PIPER 静止干跑与低速验证尚未完成，因此仍不能直接用于带电比赛运行。

## 1. 实现结构

```text
25 Hz robot loop                         GPU inference thread
────────────────────                    ──────────────────────────
next_action() → SDK执行一步
SDK确认完成 → commit(obs) ────────────→ 最新观测 + 上一块剩余动作
继续消费旧 50-step chunk                5-step Flow Matching
                                       + PiGDM guidance (β=5)
                 新 chunk 可用 ←──────── 时间对齐后原子切换
```

- `scripts/deploy/real_time_chunking.py`：异步双缓冲、延迟历史、论文软掩码、时间对齐和安全停止。
- `scripts/deploy/rtc_client_adapter.py`：连接官方 `WebsocketClientPolicy` 的轻量适配器。
- `patches/lingbot-vla-v2/rtc_flow_matching.patch`：修改 LingBot 服务端和采样器，使其接收归一化动作约束并在每个 Flow-Matching 步执行 VJP 引导。
- `configs/deploy_rtc.yaml`：GOAI 起始参数；`H=50`、25 Hz、15-step 最短执行周期、初始延迟估计10步、`β=5`。

动作约束必须位于模型的归一化空间。服务端同时返回物理动作和 `_normalized_actions`；客户端仅把后者送回 RTC 引导，避免用关节单位直接约束模型空间。

## 2. 安装上游补丁

在 `lingbot-vla-v2` 仓库根目录执行：

```bash
git apply --check /path/to/Final_GOAI/patches/lingbot-vla-v2/rtc_flow_matching.patch
git apply /path/to/Final_GOAI/patches/lingbot-vla-v2/rtc_flow_matching.patch
```

补丁以 LingBot-VLA 2.0 commit `bc643d7` 验证。若上游版本变化导致 `--check` 失败，禁止强行应用，应重新审计采样器。

## 3. 启动模型服务

```bash
cd /path/to/lingbot-vla-v2
export MODEL_PATH=/path/to/global_step_8884/hf_ckpt
bash /path/to/Final_GOAI/scripts/deploy/start_lingbot_vla_v2_rtc_server.sh
```

RTC需要对noisy action计算vector-Jacobian product。当前启动脚本关闭 `torch.compile` 并默认使用5步 Flow-Matching；RTX 6000D 实测表明10步RTC无法满足1.4秒缓冲，而5步能够满足。

## 4. 接入机器人循环

以下只描述接口顺序，不会自行发送机械臂命令：

```python
from deploy.websocket_client_policy import WebsocketClientPolicy
from scripts.deploy.rtc_client_adapter import LingBotRTCClient

policy = WebsocketClientPolicy(host="MODEL_SERVER_IP", port=8006)
rtc = LingBotRTCClient(policy)
rtc.start(read_observation())       # 首块在机器人动作前同步生成

while task_running:
    action = rtc.next_action()
    send_one_checked_action(action) # 关节限位、速度限制、急停必须在此层
    observation = read_observation_after_sdk_ack()
    rtc.commit(observation)         # 只有 SDK 确认消费后才能 commit

rtc.stop()
```

`next_action()` 和 `commit()` 不得合并：RTC 必须按“实际执行成功”的步数对齐，而不是按网络请求次数推测。若抛出 `RTCPlanExhausted`，控制层必须进入安全停止，不能重复发送最后一个关节增量。

## 5. 上真机前四道门

1. **延迟注入**：在 0/100/200/300/400 ms 注入下跑离线完整 episode，确认无停顿、无错位、无 NaN/Inf。
2. **GPU 实测**：在 RTX 6000D 上记录首块延迟、RTC 引导延迟、P50/P95/P99 和显存；要求 P99 小于 `(50-15)/25 = 1.4 s`，并留出网络抖动余量。
3. **静止干跑**：模型、相机、网络、控制循环全部运行，但电机不使能，核对每次 commit、动作索引和时间戳。
4. **低速真机**：获得明确运动授权后，先单臂空载，再双臂空载，最后按六任务逐项放开；全过程启用关节限位、速度限制、碰撞区和人工急停。

RTC 解决的是推理期间不停顿和跨块连续性，不替代机器人 SDK 的限位、低通控制、CAN 看门狗、碰撞保护或任务成功判断。

## 6. 当前验证

```bash
python -m unittest tests.test_real_time_chunking -v
git apply --check patches/lingbot-vla-v2/rtc_flow_matching.patch
```

测试覆盖：延迟换算、指数软掩码、剩余计划右侧填充、推理阻塞时继续执行、返回后按实际延迟对齐，以及动作块耗尽时强制报停。

RTX 6000D（84 GB）、BF16、step8884、`H=50`、25 Hz、15步后开始异步重规划的实测结果：

| 测试 | 结果 |
| --- | --- |
| 10步RTC进程内延迟 | 平均1.909 s，P95 1.965 s，不满足1.4 s缓冲 |
| 5步RTC进程内延迟 | 平均0.984 s，P95 0.997 s，满足缓冲 |
| WebSocket + 0/100/200 ms注入 | 最大1.110/1.233/1.280 s，均满足缓冲 |
| WebSocket + 300/400 ms注入 | 最大1.488/1.479 s，超过缓冲 |
| 六任务固定验证块，5步 | MSE 0.00194，MAE 0.02400，全部有限值 |
| 六任务固定验证块，10步 | MSE 0.00241，MAE 0.02561，全部有限值 |

六任务结果只是一任务一个固定验证 episode 首帧的50步开环抽检，说明5步没有出现明显退化，不等价于真机成功率。现场L20必须重新运行同一延迟基准；当前可接受的额外端到端延迟预算约为200 ms，不能把300 ms当作安全配置。

复现实测使用：

```bash
python scripts/deploy/benchmark_rtc_gpu.py --help
python scripts/deploy/benchmark_rtc_websocket.py --help
python scripts/deploy/benchmark_denoising_quality_websocket.py --help
```
