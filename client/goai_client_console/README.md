# HUST_HRT_GOAI 双PIPERX 控制台

Ubuntu 22.04 客户端可视化壳子。当前版本默认使用模拟数据，不连接 CAN、不下发机械臂动作。

## 安装与启动

```bash
cd goai_client_console
chmod +x install_ubuntu22.sh run.sh
./install_ubuntu22.sh
./run.sh
```

也可以直接运行：

## 明天现场启动顺序

### 1. 放置 LingBot 客户端代码

控制电脑需要有与服务器匹配的固定版本仓库：

```bash
git clone https://github.com/Robbyant/lingbot-vla-v2.git
cd lingbot-vla-v2
git checkout bc643d74a0127fab8788da993b261d4d64101138
cd ../goai_client_console
```

修改 `client_config.json` 的 `lingbot_repo`，也可以设置：

```bash
export LINGBOT_VLA_REPO=/绝对路径/lingbot-vla-v2
```

### 2. 建立 SSH 隧道

另开一个终端：

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 8006:127.0.0.1:8006 \
  root@47.97.37.69
```

保持该终端运行。不要把密码写进脚本；正式使用 SSH 密钥。

### 3. 命令行静止预热

```bash
./.venv/bin/python server_preflight.py \
  --host 127.0.0.1 \
  --port 8006 \
  --prompt "Fill the pen holder"
```

成功标准：

- TCP、WebSocket、服务端元数据和 `reset("goai_piper_x")` 均成功；
- 返回动作块严格为 `(50,14)`；
- 若启用RTC，`_normalized_actions` 可以是模型内部填充维度（当前实测为`(50,55)`），不能当作物理动作；
- 所有动作都是有限数值，无 NaN/Inf；
- 输出明确显示 `hardware_output: false`。

### 4. 启动界面

```bash
./run.sh
```

点击“真实服务端预热”会执行同一项只读测试，并把结果写到运行日志。

### 5. 三路相机检查

连接相机后执行：

```bash
v4l2-ctl --list-devices
./.venv/bin/python camera_preflight.py
```

程序会在 `camera_preflight/` 保存三张现场帧。必须人工确认顶部、左腕、右腕没有接反，并检查旋转方向、遮挡和曝光。

## 当前能力

- 三路模拟相机画面与帧率状态
- 双臂 14 维状态展示
- 推理服务器、RTC、CAN、相机、看门狗状态卡片
- 推理延迟、P95/P99、动作块进度和控制频率
- RTC 双轨时间线：当前块实际执行、下一块请求状态/耗时、返回步和切换点
- 真实服务端静止预热、元数据读取和 `(50,14)` 校验
- 只读检查、静止干跑、模拟任务和立即停止状态机
- 日志显示和保存
- 所有危险动作入口默认关闭；窗口明确显示 `MOCK / NO HARDWARE OUTPUT`

## 配置项

`client_config.json` 中需要现场确认：

- `lingbot_repo`：LingBot-VLA 2.0 仓库路径；
- `server.host/port`：SSH 隧道模式保持 `127.0.0.1:8006`；
- `camera_keys/state_key/prompt_key`：必须与服务器的 `goai_piper_x.yaml` 一致；
- `cameras`：顶部、左腕、右腕摄像头编号；
- `hardware_output_enabled`：在机械臂适配完成前必须保持 `false`。

## 后续接真机的替换点

只需要替换 `MockBackend`，保持以下 Qt 信号不变：

- `telemetry(dict)`：14 维关节/夹爪、延迟、chunk、设备状态
- `frame_ready(str, QImage)`：三路相机帧
- `log(str, str)`：日志等级和内容
- `mode_changed(str)`：当前运行阶段

接入真机前必须补齐 PIPER SDK/CAN 的单位、方向、限位、控制模式、看门狗和急停逻辑。软件停止按钮不能替代物理急停。

## 当前安全边界

已经完成的是 GUI、模拟 RTC、SSH 隧道模板和真实服务端静止预热。尚未实现且不会假装实现：

- PIPER X SDK/CAN 连接；
- 真实机械臂状态读取；
- 真实动作下发与 SDK 消费回执；
- 关节单位、方向、限位和碰撞检查；
- 物理急停联锁。

因此“真实服务端预热”返回的动作只做检查和显示，绝不会发送给机械臂。

## 已完成真实服务端冒烟测试

通过SSH隧道实测：物理动作`(50,14)`，RTC内部归一化动作`(50,55)`；第15步发出后台请求、第39步返回，新块从第24步对齐接入，最终状态为`E2E_RTC_SMOKE_OK`。本次RTC请求约960 ms，低于1.4秒动作缓存窗口但余量有限，现场仍需进行多轮P95/P99测试。
