"""Render the explicitly historical 500-frame checkpoint screening table."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "evaluation" / "model_selection_dashboard.png"
ROWS = [
    ("2221", "0.50", "0.02019", "0.06971", "0.04977", "screened"),
    ("3332", "0.75", "0.01828", "0.06517", "0.04557", "screened"),
    ("4442", "1.00", "0.01687", "0.06153", "0.04188", "screened"),
    ("5553", "1.25", "0.01543", "0.05835", "0.03893", "screened"),
    ("6663", "1.50", "0.01535", "0.05668", "0.03781", "lowest screening MSE"),
    ("7774", "1.75", "0.01603", "0.05637", "0.03605", "screened"),
    ("8884", "2.00", "0.01588", "0.05579", "0.03528", "lowest MAE / velocity"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / filename), size)


canvas = Image.new("RGB", (1500, 900), "#0d1117")
draw = ImageDraw.Draw(canvas)
draw.text((70, 42), "GOAI Historical 500-Frame Screening", fill="#f0f6fc", font=font(34, True))
draw.text(
    (70, 91),
    "30 test-split episodes · first 500 frames only · historical record, not final selection",
    fill="#8b949e",
    font=font(19),
)

box = (55, 135, 1445, 745)
draw.rounded_rectangle(box, radius=16, fill="#161b22", outline="#30363d", width=2)
columns = [80, 260, 420, 640, 850, 1080]
headers = ["Checkpoint", "Epoch", "Screening MSE", "Screening MAE", "Velocity RMSE", "Historical result"]
for x, header in zip(columns, headers):
    draw.text((x, 165), header, fill="#f0f6fc", font=font(18, True))
draw.line((75, 210, 1425, 210), fill="#30363d", width=2)

for index, row in enumerate(ROWS):
    y = 238 + index * 70
    color = "#d2a8ff" if row[0] == "8884" else "#3fb950" if row[0] == "6663" else "#c9d1d9"
    if row[0] in {"6663", "8884"}:
        fill = "#1b3a29" if row[0] == "6663" else "#292144"
        draw.rounded_rectangle((70, y - 12, 1430, y + 43), radius=10, fill=fill)
    values = row[:5] + (row[5],)
    for x, value in zip(columns, values):
        draw.text((x, y), value, fill=color, font=font(17, row[0] in {"6663", "8884"}))

draw.text((70, 785), "Final selection: global_step_8884 (2.00 epoch)", fill="#58a6ff", font=font(25, True))
draw.text(
    (70, 830),
    "Selected later on 60 full validation episodes; confirmed once on 30 full test episodes.",
    fill="#c9d1d9",
    font=font(18),
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUTPUT, optimize=True)
