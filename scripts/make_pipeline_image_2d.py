"""scripts/make_pipeline_image_2d.py
Render the UPDATED (2D-ball, 3D-removed) pipeline profile as a dark, color-coded PNG.
Saved to docs/pipeline_profile_2d.png.

What changed vs the old profile (docs/pipeline_profile.png):
  * 3D ball work removed  -> no Triangulation / 3D fusion, no Physics EKF + RTS, no dual-cam.
  * Ball detector is now a HEATMAP net at NATIVE 1280p (FP16/TensorRT target) -- the old
    4K->512x288 downscale is gone (no runtime downscaling).
  * NEW drop-on-full latest-frame ingestion ring (core/ball_stream.py) keeps latency bounded.
  * Output is tracking DATA to disk (Orin Nano has no NVENC; overlay video is rendered offline).

HONESTY: grey "Measured" text = built & measured on the dev laptop. BLUE text = DESIGN
target, not built yet (the heatmap@1280p + TensorRT path). We do not fake timings.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---- palette ----
BG = "#15151f"; HDR = "#2e2e46"; ROW_A = "#20202f"; ROW_B = "#1a1a27"
TXT = "#ebebf2"; SUB = "#a0a0b8"; DESIGN = "#7fd3ff"
STRAIN = {"HIGH": "#e74c3c", "MED": "#f1c40f", "LOWMED": "#bcd442", "LOW": "#2ecc71"}
MODE = {"P": ("#3498db", "P"), "B": ("#e67e22", "B"), "S": ("#7f8c8d", "◆")}

# n, stage, mode, resolution, device, measured, strain-label, strain-key, status(m=measured/d=design)
ROWS = [
    ("1",  "Decode  (H.265 4K, ×2 cams)",                  "S", "3840×2160 @20","GPU-NVDEC / CPU(dev)", "2×4K@20 fits*",   "MED",     "MED",    "m"),
    ("2",  "Drop-on-full latest-frame ring  (NEW)",        "S", "1 slot",      "CPU",          "~6 ms age · 0 drift",     "LOW",     "LOW",    "m"),
    ("3",  "To GPU  (zero-copy NVMM / dev copy)",          "S", "~3 MB NV12",  "NVMM / CPU→GPU", "—",                     "LOW–MED", "LOWMED", "m"),
    ("4",  "YOLO11s-pose INFERENCE (players)",             "P", "1280×736",    "GPU",          "30.6 ms",                 "HIGH",    "HIGH",   "m"),
    ("5",  "ROI filter (point-in-polygon)",                "P", "coords",      "CPU",          "<0.1 ms",                 "LOW",     "LOW",    "m"),
    ("6",  "ByteTrack (Kalman + IoU)",                     "P", "coords",      "CPU",          "~0.3 ms",                 "LOW",     "LOW",    "m"),
    ("7",  "ReID (HSV hist + Hungarian)",                  "P", "small crops", "CPU",          "1–3 ms",                  "LOW–MED", "LOWMED", "m"),
    ("8",  "Coarse: 4K→~1280 full frame (coverage)",       "B", "→1280 wide",   "GPU",          "design",                 "LOW",     "LOW",    "d"),
    ("9",  "Ball HEATMAP INFERENCE  (FP16 / TensorRT)",    "B", "coarse + crop","GPU",          "target real-time @20",   "HIGH",    "HIGH",   "d"),
    ("10", "Heatmap decode (argmax + sub-px offset)",      "B", "heatmap",      "GPU/CPU",      "design",                 "LOW",     "LOW",    "d"),
    ("11", "Native 4K crop refine (sub-px precision)",     "B", "512×512",      "GPU",          "design",                 "LOW",     "LOW",    "d"),
    ("12", "Yellow gate (HSV patch)",                      "B", "24 px patch", "CPU",          "0.12 ms",                 "LOW",     "LOW",    "m"),
    ("13", "Stationary suppressor",                        "B", "point math",  "CPU",          "0.02 ms",                 "LOW",     "LOW",    "m"),
    ("14", "Ball Kalman tracker",                          "B", "point math",  "CPU",          "0.24 ms",                 "LOW",     "LOW",    "m"),
    ("15", "Ball events (bounce / hit)",                   "B", "point math",  "CPU",          "<0.1 ms",                 "LOW",     "LOW",    "m"),
    ("16", "Homography → court metres",                    "S", "point math",  "CPU",          "<0.1 ms",                 "LOW",     "LOW",    "m"),
    ("17", "Output: tracking data → disk (JSONL/Parquet)", "S", "records",     "CPU",          "design (light)",          "LOW",     "LOW",    "d"),
    ("18", "Overlay + encode  (OFFLINE — no NVENC)",       "S", "1080p",       "CPU x264",     "offline only",            "MED",     "MED",    "m"),
]

# columns: (header, x0, x1, align)
COLS = [("#", 0, 4, "c"), ("Stage", 4, 33, "l"), ("Mode", 33, 40, "c"),
        ("Resolution", 40, 55, "c"), ("Device", 55, 70, "c"),
        ("Measured", 70, 84, "c"), ("Strain", 84, 100, "l")]

n = len(ROWS)
fig, ax = plt.subplots(figsize=(15.5, 13.5))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
ax.set_xlim(0, 100); ax.set_ylim(-7.5, n + 5.5); ax.axis("off")


def cx(c):  # center x of a column
    return (c[1] + c[2]) / 2


# ---- title + explanation ----
ax.text(0, n + 4.6, "Padel CV — 2D Ball Pipeline Profile", fontsize=24, fontweight="bold", color=TXT)
ax.text(0, n + 3.5, "Target: Jetson Orin Nano 8GB  ·  2× 4K@20 cameras  ·  measured on RTX 4050 Laptop (6 GB) / Ryzen 7 7445HS  ·  3D removed",
        fontsize=11.5, color=SUB)
ax.text(0, n + 2.2,
        "MODE = which command runs the stage:   P = players (--fuse)     B = ball (2D live / --ball-eval)     ◆ = shared by both.",
        fontsize=11, color=TXT)
ax.text(0, n + 1.3,
        "Only the two neural nets (#4 pose, #9 ball heatmap) are GPU-heavy. Cameras are 4K@20 → ball net = COARSE downscale + NATIVE 4K CROP refine.  "
        "grey = measured · blue = design target.",
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
    nn, stage, mode, res, dev, ms, slab, skey, status = r
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
    ax.text(cx(COLS[4]), y + 0.5, dev, fontsize=9.3, color=TXT, va="center", ha="center")
    # measured/design text -- BLUE when it is a design target, not a real measurement
    ax.text(cx(COLS[5]), y + 0.5, ms, fontsize=9.6, color=DESIGN if status == "d" else SUB,
            va="center", ha="center", fontstyle="italic" if status == "d" else "normal")
    # strain dot + label
    ax.scatter([85.2], [y + 0.5], s=130, c=STRAIN[skey], edgecolors="none", zorder=5)
    ax.text(86.6, y + 0.5, slab, fontsize=9.6, color=TXT, va="center", ha="left")

# ---- footnotes ----
ax.text(0, -0.6, "* Orin Nano H.265 decode does 2× 4K@30, so 2× 4K@20 fits.  Cameras cap at ~20 fps → 60 fps not achievable; target = real-time @20 fps.   "
                 "3D removed: triangulation, EKF, dual-cam.",
        fontsize=9, color=SUB)
ax.text(0, -1.25, "No NVENC on Orin Nano → production emits tracking DATA to disk; the overlay video is rendered OFFLINE from data + footage.",
        fontsize=9, color=SUB)

# ---- legend (strain + status) ----
ax.text(0, -2.5, "STRAIN", fontsize=11, fontweight="bold", color=TXT)
for k, lab, xx in [("HIGH", "HIGH  (GPU neural net)", 9),
                   ("MED", "MED  (decode, offline encode)", 34),
                   ("LOW", "LOW  (CPU point-math, ~free)", 64)]:
    ax.scatter([xx - 1.5], [-2.5], s=130, c=STRAIN[k], edgecolors="none")
    ax.text(xx, -2.5, lab, fontsize=9.6, color=SUB, va="center", ha="left")
ax.text(89, -2.5, "blue = design", fontsize=9.6, color=DESIGN, va="center", ha="left", fontstyle="italic")

# ---- status / targets box ----
ax.add_patch(FancyBboxPatch((0, -7.0), 100, 4.0, boxstyle="round,pad=0.1,rounding_size=0.4",
                            facecolor="#23233a", edgecolor="#3a3a5a", linewidth=1))
ax.text(2, -3.5, "STATUS & TARGETS   (production target = Jetson Orin Nano 8GB)", fontsize=12, fontweight="bold", color=TXT)
tp = [("Decode 2×4K@20 + drop-on-full ring", "measured: ~99 fps decode · latency ~6 ms", True),
      ("Ball CPU post (track/events/gate/suppress)", "measured: < 1 ms total", True),
      ("Player pose (YOLO11s) on dev laptop", "30.6 ms  (~33 fps)", True),
      ("Ball heatmap (coarse + native crop, FP16)", "DESIGN — target real-time @20 fps, < 50 ms", False)]
for j, (name, val, ok) in enumerate(tp):
    yy = -4.4 - j * 0.5
    mark = "✓" if ok else "◇"
    col = STRAIN["LOW"] if ok else DESIGN
    ax.text(3, yy, mark, fontsize=11, color=col, va="center", ha="center", fontweight="bold")
    ax.text(5, yy, name, fontsize=10.0, color=TXT, va="center", ha="left")
    ax.text(45, yy, val, fontsize=9.6, color=col, va="center", ha="left", fontweight="bold")
ax.text(3, -6.45, "⚠", fontsize=11, color=STRAIN["HIGH"], va="center", ha="center", fontweight="bold")
ax.text(5, -6.45, "Camera ceiling = 20 fps (4K@20 cams)", fontsize=10.0, color=TXT, va="center", ha="left")
ax.text(45, -6.45, "60 fps NOT achievable — target real-time @20 fps", fontsize=9.6, color=STRAIN["HIGH"], va="center", ha="left", fontweight="bold")

plt.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.01)
import os
os.makedirs("docs", exist_ok=True)
out = "docs/pipeline_profile_2d.png"
plt.savefig(out, dpi=130, facecolor=BG)
print("saved", out)
