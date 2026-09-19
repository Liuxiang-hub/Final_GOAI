#!/usr/bin/env python3
"""Safe visual shell for the GOAI dual-PIPER client.

The execution backend is intentionally simulation-only and never opens CAN devices.
Camera previews are read from the existing local xrobot service over HTTP.
"""

from __future__ import annotations

import math
import random
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


TASKS = [
    "笔筒装笔",
    "堆叠碗具",
    "插入充电器",
    "按压按钮",
    "物体分类",
    "折叠毛巾",
]

TASK_PROMPTS = {
    "笔筒装笔": "Fill the pen holder",
    "堆叠碗具": "Stack the bowls",
    "插入充电器": "Insert the charger",
    "按压按钮": "Press the button",
    "物体分类": "Classify the objects",
    "折叠毛巾": "Fold the towel",
}

JOINT_NAMES = [
    "L-J1", "L-J2", "L-J3", "L-J4", "L-J5", "L-J6", "L-Gripper",
    "R-J1", "R-J2", "R-J3", "R-J4", "R-J5", "R-J6", "R-Gripper",
]


class MockBackend(QObject):
    telemetry = Signal(dict)
    frame_ready = Signal(str, QImage)
    log = Signal(str, str)
    mode_changed = Signal(str)

    MODES = {"IDLE", "READ_ONLY", "DRY_RUN", "RUNNING", "STOPPED"}

    def __init__(self) -> None:
        super().__init__()
        self.mode = "IDLE"
        self.tick = 0
        self.chunk_step = 0
        self.chunk_id = 1
        self.last_actual_switch = None
        self.request_status = "未请求"
        self.request_ticks = 0
        self.request_elapsed_ms = 0
        self.returned_at_step = None
        self.aligned_start_step = None
        self.connected = False
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._update)
        self.timer.start()

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            return
        self.mode = mode
        if mode in {"DRY_RUN", "RUNNING"}:
            self.connected = True
        elif mode == "STOPPED":
            self.connected = False
            self.chunk_step = 0
            self.request_status = "未请求"
            self.request_ticks = 0
            self.request_elapsed_ms = 0
            self.returned_at_step = None
            self.aligned_start_step = None
        self.mode_changed.emit(mode)
        messages = {
            "READ_ONLY": "只读检查已开始；未打开任何 CAN 设备。",
            "DRY_RUN": "静止干跑已开始；生成模拟推理和动作块，不下发硬件。",
            "RUNNING": "模拟任务已开始；当前仍是 MOCK 模式，无硬件输出。",
            "STOPPED": "已停止并清空模拟动作队列。",
        }
        self.log.emit("WARN" if mode == "RUNNING" else "INFO", messages.get(mode, mode))

    def _camera_image(self, title: str, phase: float) -> QImage:
        width, height = 640, 360
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(QColor("#101722"))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        for x in range(0, width, 32):
            color = QColor.fromHsv((int(phase * 20) + x // 4) % 360, 95, 80)
            painter.setPen(QPen(color, 1))
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, 32):
            painter.setPen(QPen(QColor("#223044"), 1))
            painter.drawLine(0, y, width, y)
        cx = int(width / 2 + math.sin(phase) * 115)
        cy = int(height / 2 + math.cos(phase * 0.7) * 55)
        painter.setBrush(QColor("#27d4a8"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx - 22, cy - 22, 44, 44)
        painter.end()
        return image

    def _update(self) -> None:
        self.tick += 1
        active = self.mode in {"DRY_RUN", "RUNNING"}
        if active:
            if self.request_status == "已返回":
                # The returned plan is aligned by dropping actions that became stale
                # while inference was running. In this mock, 320 ms at 25 Hz = 8 steps.
                self.last_actual_switch = self.returned_at_step
                self.chunk_id += 1
                self.chunk_step = self.aligned_start_step or 0
                self.request_status = "未请求"
                self.request_ticks = 0
                self.request_elapsed_ms = 0
                self.returned_at_step = None
                self.aligned_start_step = None
            else:
                self.chunk_step += 1
                if self.request_status == "未请求" and self.chunk_step >= 15:
                    self.request_status = "推理中"
                    self.request_ticks = 0
                    self.request_elapsed_ms = 0
                elif self.request_status == "推理中":
                    self.request_ticks += 1
                    self.request_elapsed_ms = self.request_ticks * 40
                    if self.request_ticks >= 8:
                        self.request_status = "已返回"
                        self.returned_at_step = self.chunk_step
                        self.aligned_start_step = self.request_ticks
        phase = self.tick / 18.0
        states = (
            [round(math.sin(phase * 0.25 + i * 0.42) * 0.25, 4) for i in range(14)]
            if active else [0.0] * 14
        )
        latency = 0.0 if not self.connected else 34.0 + random.random() * 10.0
        request_status = self.request_status if active else "未请求"
        request_elapsed_ms = self.request_elapsed_ms if active else 0
        returned_at_step = self.returned_at_step if active else None
        current_action = [
            round(math.sin(phase * 0.45 + i * 0.31) * 0.08, 4) if active else 0.0
            for i in range(14)
        ]
        self.telemetry.emit({
            "states": states,
            "latency": latency,
            "p95": latency * 1.18 if latency else 0.0,
            "p99": latency * 1.35 if latency else 0.0,
            "chunk": self.chunk_step,
            "chunk_id": self.chunk_id,
            "next_chunk_id": self.chunk_id + 1,
            "executed": self.chunk_step,
            "request_status": request_status,
            "request_elapsed_ms": request_elapsed_ms,
            "returned_at_step": returned_at_step,
            "switch_step": 15,
            "actual_switch_step": self.last_actual_switch,
            "aligned_start_step": self.aligned_start_step,
            "current_action": current_action,
            "control_hz": 25.0 if active else 0.0,
            "server": self.connected,
            "rtc": active,
            "cameras": True,
            "can": False,
            "watchdog": True,
        })


class CameraPreviewClient(QObject):
    """Asynchronously reads the three cameras already owned by xrobot."""

    frame_ready = Signal(str, QImage)
    status_changed = Signal(bool, str)
    log = Signal(str, str)

    ROLE_TO_KEY = {
        "head": "cam_high",
        "left_wrist": "cam_left_wrist",
        "right_wrist": "cam_right_wrist",
    }

    def __init__(self, base_url: str = "http://127.0.0.1:19200") -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.manager = QNetworkAccessManager(self)
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.poll)
        self.inflight: set[str] = set()
        self.healthy_roles: set[str] = set()
        self.last_reported: bool | None = None

    def start(self) -> None:
        self.poll()
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()

    def poll(self) -> None:
        for role in self.ROLE_TO_KEY:
            if role in self.inflight:
                continue
            self.inflight.add(role)
            request = QNetworkRequest(QUrl(f"{self.base_url}/v1/preview/{role}.jpg"))
            request.setTransferTimeout(1000)
            reply = self.manager.get(request)
            reply.finished.connect(lambda role=role, reply=reply: self._finished(role, reply))

    def _finished(self, role: str, reply: QNetworkReply) -> None:
        self.inflight.discard(role)
        ok = reply.error() == QNetworkReply.NoError
        if ok:
            image = QImage.fromData(bytes(reply.readAll()), "JPG")
            ok = not image.isNull()
            if ok:
                self.healthy_roles.add(role)
                self.frame_ready.emit(self.ROLE_TO_KEY[role], image)
        else:
            self.healthy_roles.discard(role)
        reply.deleteLater()

        all_ok = len(self.healthy_roles) == len(self.ROLE_TO_KEY)
        if all_ok != self.last_reported:
            self.last_reported = all_ok
            detail = "真实三路" if all_ok else f"{len(self.healthy_roles)}/3 路"
            self.status_changed.emit(all_ok, detail)
            level = "INFO" if all_ok else "WARN"
            self.log.emit(level, f"相机预览状态：{detail}；只读取 xrobot 预览，不抢占 USB 设备。")


class StatusPill(QLabel):
    def __init__(self, name: str) -> None:
        super().__init__(f"● {name}  UNKNOWN")
        self.name = name
        self.setMinimumHeight(34)
        self.setAlignment(Qt.AlignCenter)
        self.set_state(None)

    def set_state(self, ok: bool | None, detail: str = "") -> None:
        color = "#718096" if ok is None else ("#28d7a1" if ok else "#ff637d")
        state = "UNKNOWN" if ok is None else ("OK" if ok else "OFF")
        state_cn = {"UNKNOWN": "未知", "OK": "正常", "OFF": "未连接"}[state]
        self.setText(f"● {self.name}  {detail or state_cn}")
        self.setStyleSheet(
            f"QLabel {{ color:{color}; background:#151f2d; border:1px solid #29374a; "
            "border-radius:7px; padding:6px 10px; font-weight:700; }"
        )


class CameraPanel(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("cameraPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        header = QHBoxLayout()
        name = QLabel(title)
        name.setStyleSheet("font-weight:700;color:#dce8f2")
        badge = QLabel("实时预览 · 640×480 · 5 FPS")
        badge.setStyleSheet("color:#09131d;background:#f6c85f;border-radius:5px;padding:2px 7px;font-weight:800")
        header.addWidget(name)
        header.addStretch()
        header.addWidget(badge)
        self.image = QLabel("等待视频帧")
        self.image.setMinimumSize(320, 180)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setStyleSheet("background:#0b111a;color:#718096;border-radius:6px")
        layout.addLayout(header)
        layout.addWidget(self.image, 1)

    def set_image(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image).scaled(
            self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image.setPixmap(pixmap)


class ActionTimeline(QWidget):
    """Compact RTC timeline: execution and inference are shown on separate lanes."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(182)
        self.current_id = 1
        self.next_id = 2
        self.executed = 0
        self.request_status = "未请求"
        self.request_elapsed_ms = 0
        self.returned_at_step = None
        self.switch_step = 15
        self.actual_switch_step = None
        self.aligned_start_step = None

    def update_state(self, data: dict) -> None:
        self.current_id = data.get("chunk_id", 1)
        self.next_id = data.get("next_chunk_id", self.current_id + 1)
        self.executed = data.get("executed", 0)
        self.request_status = data.get("request_status", "未请求")
        self.request_elapsed_ms = data.get("request_elapsed_ms", 0)
        self.returned_at_step = data.get("returned_at_step")
        self.switch_step = data.get("switch_step", 15)
        self.actual_switch_step = data.get("actual_switch_step")
        self.aligned_start_step = data.get("aligned_start_step")
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101722"))
        left, right = 116, 18
        usable = max(100, self.width() - left - right)
        cell = usable / 50.0

        request_summary = f"下一块 A{self.next_id:03d}：{self.request_status}"
        if self.request_status == "推理中":
            request_summary += f"  {self.request_elapsed_ms} ms"
        elif self.request_status == "已返回":
            request_summary += f"  (50,14) / 返回于旧块第 {self.returned_at_step} 步"
        painter.setPen(QColor("#58b9ff"))
        painter.setFont(QFont(QApplication.font().family(), 9, QFont.Bold))
        painter.drawText(left, 23, request_summary)
        lanes = (
            (52, f"执行块 A{self.current_id:03d}", self.executed, QColor("#27d4a8"), "execution"),
            (104, f"请求块 A{self.next_id:03d}", 50 if self.request_status == "已返回" else 0, QColor("#58b9ff"), "request"),
        )
        painter.setFont(QFont(QApplication.font().family(), 9, QFont.Bold))
        for y, label, progress, color, lane_type in lanes:
            painter.setPen(QColor("#a9bbca"))
            painter.drawText(8, y + 14, label)
            for index in range(50):
                x = left + index * cell
                rect = (int(x + 1), y, max(2, int(cell - 2)), 20)
                painter.setPen(Qt.NoPen)
                if lane_type == "request" and self.request_status == "推理中":
                    active = index % 8 in (self.executed % 8, (self.executed + 1) % 8)
                    painter.setBrush(color if active else QColor("#1d2a39"))
                else:
                    painter.setBrush(color if index < progress else QColor("#1d2a39"))
                painter.drawRoundedRect(*rect, 2, 2)

        marker_x = int(left + self.switch_step * cell)
        painter.setPen(QPen(QColor("#f6c85f"), 2, Qt.DashLine))
        painter.drawLine(marker_x, 42, marker_x, 136)
        painter.setPen(QColor("#f6c85f"))
        painter.setFont(QFont(QApplication.font().family(), 8, QFont.Bold))
        painter.drawText(marker_x + 5, 153, f"第 {self.switch_step} 步发起后台推理")
        if self.actual_switch_step is not None:
            actual_x = int(left + min(50, self.actual_switch_step) * cell)
            painter.setPen(QPen(QColor("#ff7e9d"), 2))
            painter.drawLine(actual_x, 42, actual_x, 136)
            painter.setPen(QColor("#ff9eb3"))
            painter.drawText(max(left, self.width() - 255), 23, f"上次实际切换：旧块第 {self.actual_switch_step} 步")

        painter.setPen(QColor("#6f8498"))
        painter.drawText(left, 175, "0")
        painter.drawText(int(left + 15 * cell), 175, "15")
        painter.drawText(int(left + 30 * cell), 175, "30")
        painter.drawText(int(left + 47 * cell), 175, "50")
        painter.end()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("HUST_HRT_GOAI 双PIPERX 控制台")
        self.resize(1480, 920)
        self.backend = MockBackend()
        self.preview = CameraPreviewClient()
        self.preflight_process = QProcess(self)
        self.preflight_process.setProcessChannelMode(QProcess.MergedChannels)
        self.camera_panels: dict[str, CameraPanel] = {}
        self.status: dict[str, StatusPill] = {}
        self._build_ui()
        self._connect()
        self.preview.start()
        self.append_log("INFO", "控制台已启动：真实相机预览，执行层安全模拟，不会向硬件输出动作。")

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(12)

        top = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("HUST_HRT_GOAI 双PIPERX 控制台")
        title.setObjectName("title")
        subtitle = QLabel("真实三路相机  •  真实服务端预热  •  安全模拟执行层")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        top.addLayout(title_box)
        top.addStretch()
        self.mode_badge = QLabel("待机")
        self.mode_badge.setObjectName("modeBadge")
        safety = QLabel("安全：模拟模式 / 无硬件输出")
        safety.setObjectName("safetyBadge")
        top.addWidget(self.mode_badge)
        top.addWidget(safety)
        outer.addLayout(top)

        connection = QHBoxLayout()
        connection.addWidget(QLabel("推理服务器"))
        self.host = QLineEdit("127.0.0.1")
        self.host.setMaximumWidth(170)
        self.port = QLineEdit("8006")
        self.port.setMaximumWidth(80)
        connection.addWidget(self.host)
        connection.addWidget(QLabel(":"))
        connection.addWidget(self.port)
        connection.addSpacing(18)
        connection.addWidget(QLabel("当前任务"))
        self.task = QComboBox()
        self.task.addItems(TASKS)
        self.task.setMinimumWidth(230)
        connection.addWidget(self.task)
        connection.addStretch()
        outer.addLayout(connection)

        splitter = QSplitter(Qt.Vertical)
        cameras = QWidget()
        camera_layout = QHBoxLayout(cameras)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        for key, label in (("cam_high", "顶部相机"), ("cam_left_wrist", "左腕相机"), ("cam_right_wrist", "右腕相机")):
            panel = CameraPanel(label)
            self.camera_panels[key] = panel
            camera_layout.addWidget(panel)
        splitter.addWidget(cameras)

        lower = QWidget()
        lower_layout = QHBoxLayout(lower)
        lower_layout.setContentsMargins(0, 0, 0, 0)

        telemetry_box = QGroupBox("实时遥测")
        telemetry_layout = QVBoxLayout(telemetry_box)
        metric_grid = QGridLayout()
        self.metrics: dict[str, QLabel] = {}
        for index, (key, caption, initial) in enumerate((
            ("latency", "推理延迟", "0.0 ms"), ("p95", "延迟 P95", "0.0 ms"),
            ("p99", "延迟 P99", "0.0 ms"), ("hz", "控制频率", "0.0 Hz"),
        )):
            card = QFrame()
            card.setObjectName("metricCard")
            card_layout = QVBoxLayout(card)
            cap = QLabel(caption)
            cap.setObjectName("metricCaption")
            value = QLabel(initial)
            value.setObjectName("metricValue")
            self.metrics[key] = value
            card_layout.addWidget(cap)
            card_layout.addWidget(value)
            metric_grid.addWidget(card, 0, index)
        telemetry_layout.addLayout(metric_grid)
        self.chunk_label = QLabel("动作块进度   0 / 50")
        self.chunk = QProgressBar()
        self.chunk.setRange(0, 50)
        self.chunk.setTextVisible(False)
        telemetry_layout.addWidget(self.chunk_label)
        telemetry_layout.addWidget(self.chunk)
        self.action_timeline = ActionTimeline()
        telemetry_layout.addWidget(self.action_timeline)
        self.action_vector = QLabel("模拟动作（未发送至机械臂）：等待动作")
        self.action_vector.setWordWrap(True)
        self.action_vector.setStyleSheet(
            "color:#9fb3c5;background:#101722;border:1px solid #29374a;"
            "border-radius:6px;padding:7px;font-family:monospace"
        )
        telemetry_layout.addWidget(self.action_vector)
        status_grid = QGridLayout()
        for index, key in enumerate(("服务器", "RTC", "相机", "CAN", "看门狗")):
            pill = StatusPill(key)
            status_key = {"服务器":"server", "RTC":"rtc", "相机":"cameras", "CAN":"can", "看门狗":"watchdog"}[key]
            self.status[status_key] = pill
            status_grid.addWidget(pill, index // 3, index % 3)
        telemetry_layout.addLayout(status_grid)

        state_box = QGroupBox("模拟状态预览  •  14 维（未读取真机）")
        state_layout = QVBoxLayout(state_box)
        self.state_table = QTableWidget(2, 7)
        self.state_table.setVerticalHeaderLabels(["左臂", "右臂"])
        self.state_table.setHorizontalHeaderLabels(["J1", "J2", "J3", "J4", "J5", "J6", "GRIP"])
        self.state_table.verticalHeader().setDefaultSectionSize(34)
        self.state_table.horizontalHeader().setStretchLastSection(True)
        for row in range(2):
            for col in range(7):
                item = QTableWidgetItem("+0.0000")
                item.setTextAlignment(Qt.AlignCenter)
                self.state_table.setItem(row, col, item)
        state_layout.addWidget(self.state_table)

        left_mid = QVBoxLayout()
        left_mid.addWidget(telemetry_box)
        left_mid.addWidget(state_box)
        left_widget = QWidget()
        left_widget.setLayout(left_mid)

        log_box = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont(QApplication.font().family(), 10))
        save_log = QPushButton("保存日志")
        save_log.clicked.connect(self.save_log)
        log_layout.addWidget(self.log_view)
        log_layout.addWidget(save_log)

        lower_layout.addWidget(left_widget, 3)
        lower_layout.addWidget(log_box, 2)
        splitter.addWidget(lower)
        splitter.setSizes([430, 390])
        outer.addWidget(splitter, 1)

        buttons = QHBoxLayout()
        self.read_only = QPushButton("只读检查")
        self.server_preflight = QPushButton("真实服务端预热")
        self.dry_run = QPushButton("静止干跑")
        self.start = QPushButton("模拟任务启动")
        self.start.setObjectName("startButton")
        self.stop = QPushButton("立即停止")
        self.stop.setObjectName("stopButton")
        buttons.addWidget(self.read_only)
        buttons.addWidget(self.server_preflight)
        buttons.addWidget(self.dry_run)
        buttons.addStretch()
        buttons.addWidget(self.start)
        buttons.addWidget(self.stop)
        outer.addLayout(buttons)

        self.setCentralWidget(root)
        self.setStyleSheet(STYLE)

    def _connect(self) -> None:
        self.read_only.clicked.connect(lambda: self.backend.set_mode("READ_ONLY"))
        self.server_preflight.clicked.connect(self.start_server_preflight)
        self.dry_run.clicked.connect(lambda: self.backend.set_mode("DRY_RUN"))
        self.start.clicked.connect(self.start_mock)
        self.stop.clicked.connect(lambda: self.backend.set_mode("STOPPED"))
        self.backend.telemetry.connect(self.update_telemetry)
        self.backend.frame_ready.connect(lambda key, image: self.camera_panels[key].set_image(image))
        self.backend.log.connect(self.append_log)
        self.backend.mode_changed.connect(self.update_mode)
        self.preview.frame_ready.connect(lambda key, image: self.camera_panels[key].set_image(image))
        self.preview.status_changed.connect(lambda ok, detail: self.status["cameras"].set_state(ok, detail))
        self.preview.log.connect(self.append_log)
        self.preflight_process.readyReadStandardOutput.connect(self.read_preflight_output)
        self.preflight_process.finished.connect(self.preflight_finished)

    def start_server_preflight(self) -> None:
        if self.preflight_process.state() != QProcess.NotRunning:
            self.append_log("WARN", "服务端预热正在运行，请等待当前请求结束。")
            return
        config_path = Path(__file__).with_name("client_config.json")
        self.append_log("INFO", "开始真实服务端静止观测预热；硬件输出保持关闭。")
        self.server_preflight.setEnabled(False)
        self.preflight_process.setWorkingDirectory(str(Path(__file__).parent))
        self.preflight_process.start(
            sys.executable,
            [
                str(Path(__file__).with_name("server_preflight.py")),
                "--config", str(config_path),
                "--host", self.host.text().strip(),
                "--port", self.port.text().strip(),
                "--prompt", TASK_PROMPTS[self.task.currentText()],
            ],
        )

    def read_preflight_output(self) -> None:
        raw = bytes(self.preflight_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in raw.splitlines():
            self.append_log("INFO", "服务端｜" + line)

    def preflight_finished(self, exit_code: int, _status) -> None:
        self.server_preflight.setEnabled(True)
        if exit_code == 0:
            self.append_log("INFO", "真实服务端预热通过：动作仅校验，未发送给机械臂。")
            self.status["server"].set_state(True, "真实连接")
        else:
            self.append_log("ERROR", f"真实服务端预热失败，退出码 {exit_code}。请检查隧道和配置。")
            self.status["server"].set_state(False, "预热失败")

    def start_mock(self) -> None:
        QMessageBox.information(
            self,
            "模拟模式",
            "当前版本只启动模拟任务，不会连接 CAN，也不会向机械臂发送动作。",
        )
        self.backend.set_mode("RUNNING")

    def update_mode(self, mode: str) -> None:
        self.mode_badge.setText({
            "IDLE": "待机", "READ_ONLY": "只读检查", "DRY_RUN": "静止干跑",
            "RUNNING": "模拟运行", "STOPPED": "已停止",
        }.get(mode, mode))

    def update_telemetry(self, data: dict) -> None:
        self.metrics["latency"].setText(f"{data['latency']:.1f} ms")
        self.metrics["p95"].setText(f"{data['p95']:.1f} ms")
        self.metrics["p99"].setText(f"{data['p99']:.1f} ms")
        self.metrics["hz"].setText(f"{data['control_hz']:.1f} Hz")
        self.chunk.setValue(data["chunk"])
        self.chunk_label.setText(
            f"当前执行块 A{data['chunk_id']:03d}：第 {data['executed']} / 50 步（第15步触发请求）  "
            f"｜ 下一块 A{data['next_chunk_id']:03d}：{data['request_status']}"
            + (f" {data['request_elapsed_ms']} ms" if data['request_status'] == "推理中" else "")
            + (f"，旧块第 {data['returned_at_step']} 步返回，新块从第 {data['aligned_start_step']} 步接入"
               if data['request_status'] == "已返回" else "")
        )
        self.action_timeline.update_state(data)
        values = "  ".join(f"{name}={value:+.4f}" for name, value in zip(JOINT_NAMES, data["current_action"]))
        self.action_vector.setText(f"模拟动作（未发送至机械臂）\n{values}")
        for index, value in enumerate(data["states"]):
            self.state_table.item(index // 7, index % 7).setText(f"{value:+.4f}")
        for key in ("server", "rtc", "cameras", "can", "watchdog"):
            if key == "cameras":
                continue
            detail = "安全锁定" if key == "can" else ""
            self.status[key].set_state(bool(data[key]), detail)

    def append_log(self, level: str, message: str) -> None:
        color = {"INFO": "#8ee8ce", "WARN": "#f6c85f", "ERROR": "#ff637d"}.get(level, "#c8d5e0")
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.log_view.append(f'<span style="color:#718096">{stamp}</span> '
                             f'<b style="color:{color}">[{level}]</b> {message}')

    def save_log(self) -> None:
        default = str(Path.home() / f"goai_console_{datetime.now():%Y%m%d_%H%M%S}.log")
        filename, _ = QFileDialog.getSaveFileName(self, "保存日志", default, "Log files (*.log);;All files (*)")
        if filename:
            Path(filename).write_text(self.log_view.toPlainText(), encoding="utf-8")
            self.append_log("INFO", f"日志已保存：{filename}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.preview.stop()
        self.backend.set_mode("STOPPED")
        event.accept()


STYLE = """
QWidget { background:#0d141e; color:#c8d5e0; font-size:13px; }
QLabel#title { font-size:25px; font-weight:800; letter-spacing:1px; color:#f2f7fb; }
QLabel#subtitle { color:#6f8498; font-size:11px; }
QLabel#modeBadge { background:#17354a; color:#60d9ff; border:1px solid #2d6888; border-radius:7px; padding:8px 14px; font-weight:800; }
QLabel#safetyBadge { background:#412f16; color:#ffd369; border:1px solid #7c5b21; border-radius:7px; padding:8px 14px; font-weight:800; }
QFrame#cameraPanel, QGroupBox { background:#111b28; border:1px solid #253448; border-radius:9px; }
QGroupBox { margin-top:11px; padding-top:12px; font-weight:700; color:#dce8f2; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 5px; }
QFrame#metricCard { background:#151f2d; border:1px solid #29374a; border-radius:7px; }
QLabel#metricCaption { color:#718096; font-size:10px; font-weight:700; }
QLabel#metricValue { color:#f2f7fb; font-size:20px; font-weight:800; }
QLineEdit, QComboBox, QTextEdit, QTableWidget { background:#101722; border:1px solid #2b3b50; border-radius:6px; padding:7px; selection-background-color:#246c88; }
QHeaderView::section { background:#172333; color:#8fa4b8; border:0; border-right:1px solid #29374a; padding:5px; }
QPushButton { background:#1b2a3b; border:1px solid #344a62; border-radius:7px; padding:10px 18px; font-weight:700; }
QPushButton:hover { background:#253a50; border-color:#4b6b89; }
QPushButton#startButton { background:#124f43; border-color:#218873; color:#8ff5d8; }
QPushButton#stopButton { background:#612033; border-color:#a63c58; color:#ffb0c1; font-size:14px; }
QProgressBar { background:#101722; border:1px solid #29374a; border-radius:5px; height:9px; }
QProgressBar::chunk { background:#27d4a8; border-radius:4px; }
QSplitter::handle { background:#0d141e; height:8px; }
"""


def configure_font(app: QApplication) -> None:
    """Load a CJK font explicitly when Qt cannot discover system fonts."""
    candidates = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
    )
    for font_path in candidates:
        if not font_path.exists():
            continue
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 10))
            return
    for family in ("Noto Sans CJK SC", "Microsoft YaHei UI", "WenQuanYi Micro Hei"):
        if family in QFontDatabase.families():
            app.setFont(QFont(family, 10))
            return


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("GOAI Client Console")
    configure_font(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
