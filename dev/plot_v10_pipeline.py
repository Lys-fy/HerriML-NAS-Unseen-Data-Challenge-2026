#!/usr/bin/env python3
"""v10 decision-pipeline figure (standalone image version of the mermaid
draft in [[v10 - Design + Idea Ledger]]).

Structure: vertical flow with three big container blocks (MACRO backbone
search, MICRO within-family search, RANDOM ARM), every yes/no gate drawn as
a diamond with both outcomes, dashed = abstain/default path, search spaces
in {} notation. No evidence numbers (clean structural version).
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

fig, ax = plt.subplots(figsize=(12.5, 17))
ax.set_xlim(0, 14.2)
ax.set_ylim(-1.6, 21.2)
ax.axis("off")


def box(cx, cy, w, h, text, colors=C_PLAIN, dashed=False, fs=9, bold=False):
    fc, ec = colors
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.04,rounding_size=0.12",
                       facecolor=fc, edgecolor=ec, linewidth=1.4,
                       linestyle=(0, (5, 3)) if dashed else "-", zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def diamond(cx, cy, w, h, text, fs=8.4):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, facecolor="#ffffff", edgecolor="#333333",
                         linewidth=1.4, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            linespacing=1.2)


def container(x0, y0, x1, y1, title):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#fafaf4",
                           edgecolor="#999999", linewidth=1.6, zorder=1))
    ax.text(x0 + 0.15, y1 - 0.12, title, ha="left", va="top", fontsize=10.5,
            fontweight="bold", color="#555555", zorder=2)


def arrow(pts, dashed=False, label=None, lab_dx=0.12, lab_dy=0.0, color=None):
    c = color or ("#888888" if dashed else "#333333")
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        last = i == len(pts) - 2
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle="-|>" if last else "-",
                                    color=c, lw=1.5,
                                    linestyle=(0, (5, 3)) if dashed else "-"))
    if label:
        mx, my = pts[0]
        ex, ey = pts[1]
        ax.text((mx + ex) / 2 + lab_dx, (my + ey) / 2 + lab_dy, label,
                fontsize=8, color=c, ha="left", va="center", zorder=4,
                style="italic")


# ---- top flow -------------------------------------------------------------
box(7, 20.6, 3.6, 0.9, "unseen dataset\nmetadata + clock", bold=True)
box(7, 19.2, 4.6, 1.0, "1  MEASURE\nfingerprint: corr-decay on raw tensors (CPU)", C_MEAS)
box(7, 17.8, 4.6, 1.0, "2  AUG PROBE  two-arm A/B\naug in {on + mixup, off}", C_MEAS)
diamond(7, 16.3, 3.8, 1.3, "aug arm beats\nno-aug arm?")
box(4.0, 15.1, 2.8, 0.7, "aug := on + mixup")
box(10.0, 15.1, 2.6, 0.7, "aug := off", C_DEF, dashed=True)

arrow([(7, 20.15), (7, 19.7)])
arrow([(7, 18.7), (7, 18.3)])
arrow([(7, 17.3), (7, 16.95)])
arrow([(5.7, 16.05), (4.0, 15.45)], label="yes", lab_dx=-0.7, lab_dy=0.25)
arrow([(8.3, 16.05), (10.0, 15.45)], dashed=True, label="no", lab_dx=0.25, lab_dy=0.25)

# ---- MACRO container ------------------------------------------------------
container(0.7, 10.6, 13.0, 14.45, "MACRO SEARCH    backbone in {conv, tcn1d, tcn2d}")
diamond(7, 13.5, 4.0, 1.4, "encoded data?\n(near-zero local\ncorrelation)")
box(3.0, 12.0, 3.6, 1.15, "skip the bake-off\nbackbone := TCN-2d\nsaved time -> training", C_DEF, dashed=True)
box(10.4, 12.3, 4.2, 1.0, "BACKBONE GATE\nshort-train {conv, tcn1d, tcn2d}", C_GATE)
diamond(10.4, 11.5, 3.4, 1.05, "leader margin\n> noise?")
box(7.4, 11.0, 2.7, 0.7, "backbone := leader")
box(12.6, 10.95, 2.6, 0.85, "backbone := conv\n(encoded default)", C_DEF, dashed=True)

arrow([(4.0, 14.75), (5.4, 14.1)])           # aug-on chip -> DENC
arrow([(10.0, 14.75), (8.6, 14.1)], dashed=True)  # aug-off chip -> DENC
arrow([(5.0, 13.5), (3.0, 12.58)], dashed=True, label="no: image-like", lab_dx=-2.3, lab_dy=0.3)
arrow([(9.0, 13.5), (10.4, 12.8)], label="yes: encoded", lab_dx=0.15, lab_dy=0.3)
arrow([(10.4, 11.8), (10.4, 12.02)])  # gate -> dlead (short)
arrow([(8.7, 11.5), (7.4, 11.35)], label="yes", lab_dx=-0.5, lab_dy=0.3)
arrow([(12.1, 11.5), (12.6, 11.38)], dashed=True, label="no", lab_dx=-0.2, lab_dy=0.35)

# ---- MICRO container ------------------------------------------------------
container(0.7, 6.0, 13.0, 10.2, "MICRO SEARCH    within family:  D in {5..8},  W in {16, 24, 32, 48}")
diamond(7, 9.3, 4.0, 1.3, "landscape steep?\nearly spreads\n> noise?")
box(3.0, 7.9, 3.4, 1.1, "skip micro search\narch := default D=6\nslice -> training", C_DEF, dashed=True)
box(10.4, 8.1, 4.4, 1.25, "depth-grow / width-prune\nincumbent = default D=6\ncandidates ranked by TSE\n(within-family only)", C_GATE, fs=8.5)
diamond(10.4, 6.95, 3.6, 1.15, "TSE + GATE\nbeats incumbent > noise\nAND verified?", fs=7.8)
box(7.3, 6.45, 2.9, 0.7, "adopt searched arch")
box(12.75, 6.45, 2.3, 0.7, "keep default D=6", C_DEF, dashed=True)

arrow([(3.0, 11.4), (3.0, 10.4), (5.6, 9.6)], dashed=True)     # skip-bakeoff -> dsteep
arrow([(7.4, 10.65), (7.2, 9.95)])                              # leader -> dsteep
arrow([(12.6, 10.5), (12.0, 10.1), (8.6, 9.55)], dashed=True)  # conv-default -> dsteep
arrow([(5.0, 9.3), (3.0, 8.47)], dashed=True, label="no: flat", lab_dx=-1.5, lab_dy=0.35)
arrow([(9.0, 9.3), (10.4, 8.73)], label="yes", lab_dx=0.1, lab_dy=0.3)
arrow([(10.4, 7.47), (10.4, 7.53)])
arrow([(8.6, 6.95), (7.3, 6.82)], label="yes", lab_dx=-0.5, lab_dy=0.3)
arrow([(12.2, 6.95), (12.75, 6.82)], dashed=True, label="no", lab_dx=-0.3, lab_dy=0.35)

# ---- RANDOM ARM container -------------------------------------------------
container(0.7, 1.6, 13.0, 5.6, "RANDOM ARM    arch ~ {11-op wide vocab}  x  D in {3..12}  x  W in {16, 24, 32, 48}")
diamond(7, 4.6, 4.0, 1.25, "encoded AND\nbudget remains?")
box(10.4, 3.5, 4.0, 1.0, "sample K wide archs,\nshort-train each", C_GATE)
diamond(10.4, 2.5, 3.2, 1.0, "clearly >\nincumbent?", fs=8.2)
box(7.3, 2.1, 2.9, 0.7, "adopt random arch")
box(12.8, 2.1, 2.2, 0.85, "discard, keep\nincumbent", C_DEF, dashed=True)

arrow([(3.0, 7.35), (3.0, 5.4), (5.5, 4.85)], dashed=True)     # skip-micro -> drand
arrow([(7.3, 6.1), (7.1, 5.23)])                                # adopt -> drand
arrow([(12.75, 6.1), (12.2, 5.6), (8.7, 4.95)], dashed=True)   # keep-default -> drand
arrow([(9.0, 4.6), (10.4, 4.0)], label="yes", lab_dx=0.1, lab_dy=0.3)
arrow([(5.0, 4.6), (2.2, 4.0), (2.2, 0.35), (4.6, 0.0)], dashed=True,
      label="no", lab_dx=-0.85, lab_dy=0.35)
arrow([(8.8, 2.5), (7.3, 2.45)], label="yes", lab_dx=-0.6, lab_dy=0.3)
arrow([(12.0, 2.5), (12.8, 2.53)], dashed=True, label="no", lab_dx=-0.35, lab_dy=0.3)

# ---- bottom ---------------------------------------------------------------
box(7, 0.0, 5.0, 0.95, "TRAIN-TO-BUDGET\ncosine to LR=0, EMA, best checkpoint", C_TRAIN, bold=True)
box(7, -1.25, 2.6, 0.7, "predictions", bold=True)
arrow([(7.3, 1.75), (7.1, 0.5)])
arrow([(12.8, 1.68), (12.0, 0.9), (9.5, 0.25)], dashed=True)
arrow([(7, -0.48), (7, -0.9)])

# ---- decision record ------------------------------------------------------
box(13.0, 16.6, 2.2, 1.5, "decision\nrecord JSON\n(every gate:\nmeasure, margin,\nchoice, time)", C_ART, fs=8)
for y in (13.5, 9.3, 4.6):
    arrow([(13.05, y + 1.0), (13.15, 15.8)], dashed=True, color=C_ART[1])

# legend
lx, ly = 1.0, 20.8
for dy, (colors, lab, dash) in enumerate([
        (C_MEAS, "cheap measurement", False),
        (C_GATE, "gated probe / search", False),
        (C_TRAIN, "the expensive stage", False),
        (C_DEF, "abstain / default path", True)]):
    box(lx + 0.35, ly - dy * 0.55, 0.6, 0.34, "", colors, dashed=dash)
    ax.text(lx + 0.8, ly - dy * 0.55, lab, fontsize=8.5, va="center")

fig.suptitle("v10: the measure-gate-train decision pipeline (per unseen dataset)",
             fontsize=15, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.975])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
