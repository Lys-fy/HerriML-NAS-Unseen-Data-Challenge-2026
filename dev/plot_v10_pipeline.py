#!/usr/bin/env python3
"""v10 decision-pipeline figure (standalone image version of the mermaid
draft in [[v10 - Design + Idea Ledger]]).

Structure: vertical flow with three big container blocks (MACRO backbone
search, MICRO within-family search, RANDOM ARM), every yes/no gate drawn as
a diamond with both outcomes, dashed = abstain/default path, search spaces
in {} notation. No evidence numbers (clean structural version).
Layout uses a large canvas with generous spacing; the decision-record rail
sits fully outside the containers to avoid any overlap.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/notes/09 - Implementation/implementation/images/v10_pipeline.png"

C_MEAS = ("#e8f3fc", "#1f77b4")   # measurement (cheap)
C_GATE = ("#fff3cd", "#b8860b")   # gated probe / search
C_TRAIN = ("#fdecea", "#b53030")  # the expensive stage
C_DEF = ("#f2f2f2", "#888888")    # abstain / default path
C_ART = ("#f6f0fa", "#7d5ba6")    # artifact
C_PLAIN = ("#ffffff", "#444444")

fig, ax = plt.subplots(figsize=(15, 21))
ax.set_xlim(0, 21)
ax.set_ylim(0, 30.6)
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


# ---- top flow (spine at x=9) ----------------------------------------------
box(9, 29.6, 4.0, 0.9, "unseen dataset\nmetadata + clock", bold=True)
box(9, 28.2, 6.6, 1.0, "1  MEASURE\nfingerprint: corr-decay on raw tensors (CPU)", C_MEAS)
box(9, 26.8, 6.6, 1.0, "2  AUG PROBE  two-arm A/B\naug in {on + mixup, off}", C_MEAS)
diamond(9, 25.2, 4.2, 1.4, "aug arm beats\nno-aug arm?")
box(5.0, 23.9, 3.2, 0.75, "aug := on + mixup")
box(13.0, 23.9, 2.9, 0.75, "aug := off", C_DEF, dashed=True)

arrow([(9, 29.15), (9, 28.7)])
arrow([(9, 27.7), (9, 27.3)])
arrow([(9, 26.3), (9, 25.9)])
arrow([(6.9, 25.2), (5.0, 24.28)], label="yes", lab_dx=-0.85, lab_dy=0.3)
arrow([(11.1, 25.2), (13.0, 24.28)], dashed=True, label="no", lab_dx=0.35, lab_dy=0.3)

# ---- MACRO container -------------------------------------------------------
container(0.8, 16.7, 17.6, 22.9, "MACRO SEARCH      backbone in {conv, tcn1d, tcn2d}")
diamond(9, 21.3, 4.6, 1.6, "encoded data?\n(near-zero local\ncorrelation)")
box(3.7, 19.3, 4.2, 1.35, "skip the bake-off\nbackbone := TCN-2d\nsaved time -> training", C_DEF, dashed=True)
box(13.4, 19.6, 5.2, 1.1, "BACKBONE GATE\nshort-train {conv, tcn1d, tcn2d}", C_GATE)
diamond(13.4, 18.2, 3.8, 1.25, "leader margin\n> noise?")
box(9.7, 17.5, 3.1, 0.75, "backbone := leader")
box(16.0, 17.4, 3.1, 0.95, "backbone := conv\n(encoded default)", C_DEF, dashed=True)

arrow([(5.0, 23.5), (5.0, 22.6), (7.5, 21.9)])                 # aug-on -> DENC
arrow([(13.0, 23.5), (13.0, 22.6), (10.5, 21.9)], dashed=True)  # aug-off -> DENC
arrow([(6.7, 21.3), (3.7, 20.0)], dashed=True, label="no: image-like",
      lab_dx=-2.6, lab_dy=0.45)
arrow([(11.3, 21.3), (13.4, 20.17)], label="yes: encoded", lab_dx=0.3, lab_dy=0.4)
arrow([(13.4, 19.05), (13.4, 18.85)])
arrow([(11.5, 18.2), (9.7, 17.9)], label="yes", lab_dx=-0.75, lab_dy=0.35)
arrow([(15.3, 18.2), (16.0, 17.9)], dashed=True, label="no", lab_dx=-0.15, lab_dy=0.4)

# ---- MICRO container -------------------------------------------------------
container(0.8, 9.7, 17.6, 16.0, "MICRO SEARCH      within family:  D in {5..8},   W in {16, 24, 32, 48}")
diamond(9, 14.5, 4.6, 1.5, "landscape steep?\nearly spreads\n> noise?")
box(3.7, 12.4, 4.2, 1.35, "skip micro search\narch := default D=6\nslice -> training", C_DEF, dashed=True)
box(13.4, 12.9, 5.4, 1.55, "depth-grow / width-prune\nincumbent = default D=6\ncandidates ranked by TSE\n(within-family only)", C_GATE, fs=9)
diamond(13.4, 11.3, 4.3, 1.4, "TSE + GATE\nbeats incumbent\n> noise AND verified?", fs=8.6)
box(9.5, 10.5, 3.1, 0.75, "adopt searched arch")
box(16.2, 10.45, 2.7, 0.75, "keep default D=6", C_DEF, dashed=True, fs=9)

arrow([(3.7, 18.6), (3.7, 17.2), (3.7, 15.6), (7.0, 14.9)], dashed=True)
arrow([(9.7, 17.1), (9.2, 15.27)])
arrow([(16.0, 16.9), (16.0, 15.8), (11.0, 14.95)], dashed=True)
arrow([(6.7, 14.5), (3.7, 13.1)], dashed=True, label="no: flat", lab_dx=-1.9, lab_dy=0.45)
arrow([(11.3, 14.5), (13.4, 13.7)], label="yes", lab_dx=0.25, lab_dy=0.4)
arrow([(13.4, 12.12), (13.4, 12.0)])
arrow([(11.3, 11.3), (9.5, 10.9)], label="yes", lab_dx=-0.8, lab_dy=0.35)
arrow([(15.5, 11.3), (16.2, 10.85)], dashed=True, label="no", lab_dx=-0.1, lab_dy=0.4)

# ---- RANDOM ARM container ---------------------------------------------------
container(0.8, 2.9, 17.6, 9.0, "RANDOM ARM      arch ~ {11-op wide vocab}  x  D in {3..12}  x  W in {16, 24, 32, 48}")
diamond(9, 7.6, 4.6, 1.4, "encoded AND\nbudget remains?")
box(13.4, 6.2, 4.8, 1.0, "sample K wide archs,\nshort-train each", C_GATE)
diamond(13.4, 4.9, 3.7, 1.2, "clearly >\nincumbent?")
box(9.5, 4.1, 3.3, 0.75, "adopt random arch")
box(16.2, 4.05, 2.7, 0.95, "discard, keep\nincumbent", C_DEF, dashed=True, fs=9)

arrow([(3.7, 11.7), (3.7, 9.4), (3.7, 8.6), (6.9, 7.95)], dashed=True)
arrow([(9.5, 10.1), (9.1, 8.32)])
arrow([(16.2, 10.05), (16.2, 9.3), (11.1, 7.95)], dashed=True)
arrow([(11.3, 7.6), (13.4, 6.72)], label="yes", lab_dx=0.25, lab_dy=0.4)
arrow([(6.7, 7.6), (1.9, 6.6), (1.9, 1.7), (6.1, 1.45)], dashed=True,
      label="no", lab_dx=-1.1, lab_dy=0.4)
arrow([(13.4, 5.7), (13.4, 5.52)])
arrow([(11.6, 4.9), (9.5, 4.48)], label="yes", lab_dx=-0.85, lab_dy=0.35)
arrow([(15.2, 4.9), (16.2, 4.55)], dashed=True, label="no", lab_dx=-0.15, lab_dy=0.4)

# ---- bottom -----------------------------------------------------------------
box(9, 1.3, 5.8, 1.0, "TRAIN-TO-BUDGET\ncosine to LR=0, EMA, best checkpoint", C_TRAIN, bold=True)
box(9, 0.25, 2.8, 0.55, "predictions", bold=True)
arrow([(9.5, 3.7), (9.2, 1.85)])
arrow([(16.2, 3.55), (16.2, 2.4), (11.95, 1.5)], dashed=True)
arrow([(9, 0.78), (9, 0.55)])

# ---- decision record (rail fully OUTSIDE the containers) ---------------------
RAILX = 18.6
box(19.5, 1.3, 2.7, 1.7, "decision record\nJSON\n(every gate:\nmeasure, margin,\nchoice, time)", C_ART, fs=8.8)
arrow([(12.3, 28.2), (RAILX, 28.2)], dashed=True, color=C_ART[1])     # measure stub
arrow([(17.6, 19.6), (RAILX, 19.6)], dashed=True, color=C_ART[1])     # macro stub
arrow([(17.6, 12.9), (RAILX, 12.9)], dashed=True, color=C_ART[1])     # micro stub
arrow([(17.6, 6.2), (RAILX, 6.2)], dashed=True, color=C_ART[1])       # random stub
arrow([(11.9, 1.3), (18.15, 1.3)], dashed=True, color=C_ART[1])       # train stub
ax.plot([RAILX, RAILX], [28.2, 1.3], linestyle=(0, (5, 3)),
        color=C_ART[1], lw=1.6, zorder=2)

# ---- legend (top left, clear of everything) ----------------------------------
lx, ly = 1.4, 29.9
for dy, (colors, lab, dash) in enumerate([
        (C_MEAS, "cheap measurement", False),
        (C_GATE, "gated probe / search", False),
        (C_TRAIN, "the expensive stage", False),
        (C_DEF, "abstain / default path", True)]):
    box(lx + 0.4, ly - dy * 0.62, 0.7, 0.4, "", colors, dashed=dash)
    ax.text(lx + 0.95, ly - dy * 0.62, lab, fontsize=9.5, va="center")

fig.suptitle("v10: the measure-gate-train decision pipeline (per unseen dataset)",
             fontsize=17, y=0.988)
fig.tight_layout(rect=[0, 0, 1, 0.978])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
