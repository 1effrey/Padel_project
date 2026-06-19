"""scripts/make_pipeline_image.py
Render the full-pipeline profile as a dark, color-coded PNG (table + legend +
explanation) so anyone can read it at a glance. Saved to docs/pipeline_profile.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---- palette ----
BG = "#15151f"; HDR = "#2e2e46"; ROW_A = "#20202f"; ROW_B = "#1a1a27"
TXT = "#ebebf2"; SUB = "#a0a0b8"
STRAIN = {"HIGH": "#e74c3c", "MED": "#f1c40f", "LOWMED": "#bcd442", "LOW": "#2ecc71"}
MODE = {"P": ("#3498db", "P"), "B": ("#e67e22", "B"), "S": ("#7f8c8d", "◆")}

# n, stage, mode, resolution, device, measured, strain-label, strain-key
ROWS = [
    ("1",  "Decode  (decompress HEVC 4K)",        "S", "3840×2160",   "GPU-NVDEC",  "29 / 1.6 ms*", "MED · HIGH(CPU)", "MED"),
    ("2",  "Upload frame → GPU memory",            "S", "~24 MB",      "CPU→GPU",    "—",            "MED",             "MED"),
    ("3a", "YOLO resize 4K→1280 (letterbox)",      "P", "4K→1280×736", "CPU",        "part of 3b",   "MED",             "MED"),
    ("3b", "YOLO11s-pose INFERENCE (players)",     "P", "1280×736",    "GPU",        "30.6 ms",      "HIGH",            "HIGH"),
    ("4",  "ROI filter (point-in-polygon)",        "P", "4K coords",   "CPU",        "<0.1 ms",      "LOW",             "LOW"),
    ("5",  "ByteTrack (Kalman + IoU)",             "P", "4K coords",   "CPU",        "~0.3 ms",      "LOW",             "LOW"),
    ("6",  "ReID (HSV hist + Hungarian)",          "P", "small crops", "CPU",        "1–3 ms",       "LOW–MED",         "LOWMED"),
    ("7",  "Homography → metres",                  "S", "point math",  "CPU",        "<0.1 ms",      "LOW",             "LOW"),
    ("8",  "Cross-camera fusion (dedup/merge)",    "S", "point math",  "CPU",        "<0.5 ms",      "LOW",             "LOW"),
    ("9",  "TrackNet resize 4K→512×288 (×3)",      "B", "4K→512×288",  "CPU/GPU",    "part of 10",   "MED",             "MED"),
    ("10", "TrackNet INFERENCE (ball)",            "B", "512×288",     "GPU",        "~59 ms†",      "HIGH",            "HIGH"),
    ("11", "Yellow gate (HSV patch)",              "B", "24 px patch", "CPU",        "0.12 ms",      "LOW",             "LOW"),
    ("12", "Stationary suppressor  (NEW)",         "B", "point math",  "CPU",        "0.02 ms",      "LOW",             "LOW"),
    ("13", "Ball Kalman tracker",                  "B", "point math",  "CPU",        "0.24 ms",      "LOW",             "LOW"),
    ("14", "Ball events (bounce / hit)",           "B", "point math",  "CPU",        "<0.1 ms",      "LOW",             "LOW"),
    ("15", "Triangulation / 3D fusion",            "B", "point math",  "CPU",        "<0.2 ms",      "LOW",             "LOW"),
    ("16", "Physics EKF + RTS  (Phase 5, built)",  "B", "trajectory",  "CPU",        "batch/offline","LOW",             "LOW"),
    ("17", "Draw overlays + minimap",              "S", "4K canvas",   "CPU",        "few ms",       "MED",             "MED"),
    ("18", "Encode (save video)",                  "S", "4K",          "CPU (mp4v)", "only --save",  "HIGH",            "HIGH"),
    ("19", "Display window",                       "S", "1280×720",    "CPU/GPU",    "—",            "LOW–MED",         "LOWMED"),
]

# columns: (header, x0, x1, align)
COLS = [("#", 0, 4, "c"), ("Stage", 4, 33, "l"), ("Mode", 33, 40, "c"),
        ("Resolution", 40, 55, "c"), ("Device", 55, 68, "c"),
        ("Measured", 68, 82, "c"), ("Strain", 82, 100, "l")]

n = len(ROWS)
fig, ax = plt.subplots(figsize=(15.5, 13.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 100); ax.set_ylim(-7.5, n + 5.5); ax.axis("off")


def cx(c):  # center x of a column
    return (c[1] + c[2]) / 2


# ---- title + explanation ----
ax.text(0, n + 4.6, "Padel CV — Full Pipeline Profile", fontsize=24, fontweight="bold", color=TXT)
ax.text(0, n + 3.5, "Measured on RTX 4050 Laptop (6 GB) / Ryzen 7 7445HS  ·  the project runs as TWO separate modes",
        fontsize=11.5, color=SUB)
ax.text(0, n + 2.2,
        "MODE = which command runs the stage:   P = players (--fuse)     B = ball (--ball-3d / --ball-dual)     ◆ = shared by both.",
        fontsize=11, color=TXT)
ax.text(0, n + 1.3,
        "Read it this way: only the TWO neural nets (3b, 10) are GPU-heavy. Everything else is light CPU point-math — including the new suppressor.",
        fontsize=11, color=SUB)

# ---- header ----
ax.add_patch(Rectangle((0, n), 100, 1, facecolor=HDR, edgecolor="none"))
for c in COLS:
    if c[3] == "l":
        ax.text(c[1] + 0.6, n + 0.5, c[0], fontsize=12, fontweight="bold", color=TXT, va="center", ha="left")
    else:
        ax.text(cx(c), n + 0.5, c[0], fontsize=12, fontweight="bold", color=TXT, va="center", ha="center")

# ---- rows ----
for i, r in enumerate(ROWS):
    y = n - 1 - i
    ax.add_patch(Rectangle((0, y), 100, 1, facecolor=ROW_A if i % 2 == 0 else ROW_B, edgecolor="none"))
    nn, stage, mode, res, dev, ms, slab, skey = r
    ax.text(cx(COLS[0]), y + 0.5, nn, fontsize=10, color=SUB, va="center", ha="center")
    bold = "INFERENCE" in stage or "NEW" in stage
    ax.text(COLS[1][1] + 0.6, y + 0.5, stage, fontsize=10.3, color=TXT, va="center", ha="left",
            fontweight="bold" if bold else "normal")
    # mode pill
    mc, ml = MODE[mode]
    ax.add_patch(FancyBboxPatch((cx(COLS[2]) - 1.7, y + 0.2), 3.4, 0.6,
                                boxstyle="round,pad=0.02,rounding_size=0.25", facecolor=mc, edgecolor="none"))
    ax.text(cx(COLS[2]), y + 0.5, ml, fontsize=10.5, color="white", fontweight="bold", va="center", ha="center")
    ax.text(cx(COLS[3]), y + 0.5, res, fontsize=9.6, color=SUB, va="center", ha="center")
    ax.text(cx(COLS[4]), y + 0.5, dev, fontsize=9.6, color=TXT, va="center", ha="center")
    ax.text(cx(COLS[5]), y + 0.5, ms, fontsize=9.6, color=SUB, va="center", ha="center")
    # strain dot (scatter keeps it round) + label
    ax.scatter([83.2], [y + 0.5], s=130, c=STRAIN[skey], edgecolors="none", zorder=5)
    ax.text(84.6, y + 0.5, slab, fontsize=9.6, color=TXT, va="center", ha="left")

# ---- footnotes ----
ax.text(0, -0.6, "* decode: 29 ms raw, ~1.6 ms when overlapped by the threaded reader.   "
                 "† TrackNet 59 ms includes its 4K→512 resize + heat-map post-processing.",
        fontsize=9, color=SUB)

# ---- strain legend ----
ax.text(0, -1.9, "STRAIN", fontsize=11, fontweight="bold", color=TXT)
for k, lab, xx in [("HIGH", "HIGH  (GPU neural net / video encode)", 9),
                   ("MED", "MED  (decode, resize, 4K draw, copy)", 44),
                   ("LOW", "LOW  (CPU point-math, ~free)", 74)]:
    ax.scatter([xx - 1.5], [-1.9], s=130, c=STRAIN[k], edgecolors="none")
    ax.text(xx, -1.9, lab, fontsize=9.6, color=SUB, va="center", ha="left")

# ---- throughput box ----
ax.add_patch(FancyBboxPatch((0, -7.0), 100, 4.3, boxstyle="round,pad=0.1,rounding_size=0.4",
                            facecolor="#23233a", edgecolor="#3a3a5a", linewidth=1))
ax.text(2, -3.4, "THROUGHPUT on this laptop   (live bar = 20 fps)", fontsize=12, fontweight="bold", color=TXT)
tp = [("Player  single-cam (--fuse)", "36 fps", True),
      ("Player  dual-cam (--fuse)", "~15 fps (est)", False),
      ("Ball  single-cam", "17 fps", False),
      ("Ball  dual-cam (--ball-dual)", "10.7 fps", False),
      ("Players + Ball, dual  (full vision)", "~5–6 fps (est)", False)]
for j, (name, val, ok) in enumerate(tp):
    yy = -4.4 - j * 0.55
    mark = "✓" if ok else "✗"
    col = STRAIN["LOW"] if ok else STRAIN["HIGH"]
    ax.text(3, yy, mark, fontsize=11, color=col, va="center", ha="center", fontweight="bold")
    ax.text(5, yy, name, fontsize=10.2, color=TXT, va="center", ha="left")
    ax.text(46, yy, val, fontsize=10.2, color=col, va="center", ha="left", fontweight="bold")
ax.text(60, -4.4, "VRAM: YOLO 0.18 + TrackNet 0.24 GB", fontsize=10, color=SUB, ha="left")
ax.text(60, -5.0, "→ ~1.3–1.6 GB total (fits 6 GB & Jetson 8 GB)", fontsize=10, color=SUB, ha="left")
ax.text(60, -5.9, "Bottleneck = the 2 neural nets only.", fontsize=10.3, color=TXT, ha="left")
ax.text(60, -6.5, "Real-time path = TensorRT / FP16 on them.", fontsize=10.3, color="#7fd3ff", ha="left")

plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
import os
os.makedirs("docs", exist_ok=True)
out = "docs/pipeline_profile.png"
plt.savefig(out, dpi=130, facecolor=BG)
print("saved", out)
