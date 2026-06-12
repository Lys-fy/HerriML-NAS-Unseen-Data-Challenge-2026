#!/usr/bin/env python3
"""v10 decision-pipeline figure, AS BUILT in baselines/v10 (2026-06-11).

Matches the implementation: aug probe AFTER the backbone choice with
family-matched arms and the asymmetric off-margin rule (A5b + A6); micro =
depth only with the D=6 incumbent (width search and TSE dropped per
Phase A); random arm encoded-only; low-budget collapse note; decision-record
rail outside the containers. No evidence numbers (clean structural version).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/v10_pipeline.png"

C_MEAS = ("#e8f3fc", "#1f77b4")
C_GATE = ("#fff3cd", "#b8860b")
C_TRAIN = ("#fdecea", "#b53030")
C_DEF = ("#f2f2f2", "#888888")
C_ART = ("#f6f0fa", "#7d5ba6")
C_PLAIN = ("#ffffff", "#444444")

fig, ax = plt.subplots(figsize=(13.5, 22))
ax.set_xlim(0, 18.4)
ax.set_ylim(3.8, 32.4)
ax.axis("off")


def box(cx, cy, w, h, text, colors=C_PLAIN, dashed=False, fs=9.5, bold=False):
    fc, ec = colors
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.05,rounding_size=0.14",
                       facecolor=fc, edgecolor=ec, linewidth=1.5,
                       linestyle=(0, (5, 3)) if dashed else "-", zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def diamond(cx, cy, w, h, text, fs=9):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, facecolor="#ffffff", edgecolor="#333333",
                         linewidth=1.5, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            linespacing=1.25)


def container(x0, y0, x1, y1, title):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#fafaf2",
                           edgecolor="#999999", linewidth=1.8, zorder=1))
    ax.text(x0 + 0.25, y1 - 0.18, title, ha="left", va="top", fontsize=12,
            fontweight="bold", color="#555555", zorder=2)


def arrow(pts, dashed=False, label=None, lab_dx=0.15, lab_dy=0.0, color=None):
    c = color or ("#888888" if dashed else "#333333")
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        last = i == len(pts) - 2
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle="-|>" if last else "-",
                                    color=c, lw=1.6,
                                    linestyle=(0, (5, 3)) if dashed else "-"))
    if label:
        mx, my = pts[0]
        ex, ey = pts[1]
        ax.text((mx + ex) / 2 + lab_dx, (my + ey) / 2 + lab_dy, label,
                fontsize=9, color=c, ha="left", va="center", zorder=4,
                style="italic")


# ---- top ---------------------------------------------------------------------
box(9, 31.6, 4.0, 0.9, "unseen dataset\nmetadata + clock", bold=True)
box(9, 30.3, 6.8, 1.0, "1  MEASURE (DataProcessor)\nfingerprint: corr1 on raw tensors, CPU seconds", C_MEAS)
box(3.0, 30.3, 4.4, 1.45, "low-budget collapse:\n< 35 min: no gate / depth /\nrandom (encoded -> conv);\n< 15 min: no aug probe", C_DEF, dashed=True, fs=8.5)
arrow([(9, 31.15), (9, 30.8)])

# ---- MACRO -------------------------------------------------------------------
container(0.8, 23.7, 17.6, 29.4, "2  MACRO      backbone in {conv, tcn1d, tcn2d}")
diamond(9, 28.0, 4.4, 1.5, "encoded data?\ncorr1 < 0.2")
box(3.7, 26.0, 4.2, 1.5, "skip the bake-off\nbackbone := TCN-2d\n+ channel-embed stem (ch >= 8)\nsaved time -> training", C_DEF, dashed=True, fs=8.8)
box(13.4, 26.3, 5.2, 1.1, "BACKBONE GATE\nshort-train {conv, tcn1d, tcn2d}", C_GATE)
diamond(13.4, 24.95, 4.0, 1.25, "leader CLEARLY beats\nthe conv arm?\n(margin > noise)", fs=8.4)
box(9.6, 24.25, 3.1, 0.75, "backbone := leader")
box(16.2, 24.2, 2.9, 0.95, "backbone := conv\n(encoded default)", C_DEF, dashed=True, fs=9)

arrow([(9, 29.8), (9, 28.75)])
arrow([(6.8, 28.0), (3.7, 26.78)], dashed=True, label="no: image-like",
      lab_dx=-2.7, lab_dy=0.45)
arrow([(11.2, 28.0), (13.4, 26.87)], label="yes: encoded", lab_dx=0.3, lab_dy=0.4)
arrow([(13.4, 25.75), (13.4, 25.57)])
arrow([(11.4, 24.95), (9.6, 24.62)], label="yes", lab_dx=-0.75, lab_dy=0.35)
arrow([(15.4, 24.95), (16.2, 24.7)], dashed=True, label="no", lab_dx=-0.15, lab_dy=0.4)

# ---- AUG ---------------------------------------------------------------------
container(0.8, 19.0, 17.6, 23.1, "3  AUG PROBE      family-matched arms, AFTER the backbone choice")
box(9, 22.0, 6.4, 1.0, "two small arms OF THE CHOSEN FAMILY\naug-on vs aug-off", C_MEAS)
diamond(9, 20.55, 5.0, 1.4, "off wins by a CLEAR margin?\n(asymmetric: aug = regularization,\noff leads at probe horizons)", fs=8.3)
box(5.0, 19.55, 3.0, 0.7, "aug := off + mixup")
box(13.2, 19.55, 3.2, 0.7, "aug := on (metadata default)", C_DEF, dashed=True, fs=8.7)

arrow([(3.7, 25.25), (3.7, 23.6), (5.9, 22.35)], dashed=True)   # skip -> arms
arrow([(9.6, 23.87), (9.2, 22.5)])                              # leader -> arms
arrow([(16.2, 23.72), (16.2, 23.3), (12.1, 22.4)], dashed=True)  # conv default -> arms
arrow([(9, 21.5), (9, 21.25)])
arrow([(6.5, 20.55), (5.0, 19.9)], label="yes", lab_dx=-0.7, lab_dy=0.3)
arrow([(11.5, 20.55), (13.2, 19.9)], dashed=True, label="no", lab_dx=0.2, lab_dy=0.35)

# ---- MICRO -------------------------------------------------------------------
container(0.8, 13.9, 17.6, 18.6, "4  MICRO      depth only:  D in {4, 6, 8}")
box(9, 17.4, 7.0, 1.2, "short-train candidates {4, 8} vs the INCUMBENT default D=6\n(skipped for tcn1d / low budget)", C_GATE, fs=9)
diamond(9, 15.8, 4.8, 1.4, "candidate beats incumbent\n> noise AND verified\nat a longer horizon?", fs=8.5)
box(5.0, 14.7, 3.2, 0.7, "adopt searched depth")
box(13.2, 14.7, 3.0, 0.7, "keep default D=6", C_DEF, dashed=True, fs=9)

arrow([(5.0, 19.2), (5.0, 18.9), (7.4, 18.05)])
arrow([(13.2, 19.2), (13.2, 18.9), (10.7, 18.05)], dashed=True)
arrow([(9, 16.8), (9, 16.5)])
arrow([(6.6, 15.8), (5.0, 15.05)], label="yes", lab_dx=-0.7, lab_dy=0.3)
arrow([(11.4, 15.8), (13.2, 15.05)], dashed=True, label="no", lab_dx=0.2, lab_dy=0.35)

# ---- RANDOM ------------------------------------------------------------------
container(0.8, 7.7, 17.6, 13.5, "5  RANDOM ARM      arch ~ {11-op vocab}  x  D in {3..12}  x  W in {16, 24, 32, 48}")
diamond(9, 12.1, 4.6, 1.4, "encoded AND\nbudget remains?")
box(13.4, 10.7, 4.8, 1.0, "sample K=6 wide archs,\nshort-train each", C_GATE)
diamond(13.4, 9.4, 3.9, 1.2, "clearly > incumbent?\n(margin > 2x noise)", fs=8.5)
box(9.5, 8.6, 3.3, 0.75, "adopt random arch")
box(16.2, 8.55, 2.7, 0.95, "discard, keep\nincumbent", C_DEF, dashed=True, fs=9)

arrow([(5.0, 14.35), (5.0, 13.7), (7.2, 12.55)])
arrow([(13.2, 14.35), (13.2, 13.7), (10.9, 12.6)], dashed=True)
arrow([(11.3, 12.1), (13.4, 11.22)], label="yes", lab_dx=0.25, lab_dy=0.4)
arrow([(6.7, 12.1), (1.9, 11.0), (1.9, 6.2), (6.1, 5.85)], dashed=True,
      label="no", lab_dx=-1.1, lab_dy=0.4)
arrow([(13.4, 10.2), (13.4, 10.02)])
arrow([(11.6, 9.4), (9.5, 8.98)], label="yes", lab_dx=-0.85, lab_dy=0.35)
arrow([(15.3, 9.4), (16.2, 9.05)], dashed=True, label="no", lab_dx=-0.15, lab_dy=0.4)

# ---- bottom ------------------------------------------------------------------
box(9, 5.7, 5.8, 1.0, "6  TRAIN-TO-BUDGET\ncosine to LR=0, EMA, best checkpoint", C_TRAIN, bold=True)
box(9, 4.45, 2.8, 0.6, "predictions", bold=True)
arrow([(9.5, 8.2), (9.2, 6.25)])
arrow([(16.2, 8.05), (16.2, 7.0), (11.95, 5.9)], dashed=True)
arrow([(9, 5.18), (9, 4.78)])

# ---- legend ------------------------------------------------------------------
lx, ly = 14.6, 31.9
for dy, (colors, lab, dash) in enumerate([
        (C_MEAS, "cheap measurement", False),
        (C_GATE, "gated probe / search", False),
        (C_TRAIN, "the expensive stage", False),
        (C_DEF, "abstain / default path", True)]):
    box(lx + 0.4, ly - dy * 0.58, 0.7, 0.38, "", colors, dashed=dash)
    ax.text(lx + 0.95, ly - dy * 0.58, lab, fontsize=9.5, va="center")

fig.suptitle("v10 as built: the measure-gate-train decision pipeline (per unseen dataset)",
             fontsize=16, y=0.988)
fig.tight_layout(rect=[0, 0, 1, 0.978])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
