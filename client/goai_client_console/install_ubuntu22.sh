#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

sudo apt update
sudo apt install -y python3-venv libxcb-cursor0 fonts-noto-cjk libgl1 libglib2.0-0 openssh-client v4l-utils
/usr/bin/python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo "安装完成。"
echo "1) 修改 client_config.json 中的 lingbot_repo 和相机编号"
echo "2) 建立SSH隧道"
echo "3) 执行 ./.venv/bin/python server_preflight.py"
echo "4) 执行 ./run.sh"

