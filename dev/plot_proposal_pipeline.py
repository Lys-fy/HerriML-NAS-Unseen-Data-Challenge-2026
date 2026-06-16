#!/usr/bin/env python3
"""Proposal decision-pipeline figure (docs/proposal/pipeline.png).

Corrector-facing, pastel decision tree. Stage names match the proposal text
(Family selection / Augmentation probe / Structural search / Operation
exploration). Structural search covers BOTH depth and width. Committed so the
source is not lost again (the original lived in /tmp).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs/proposal/pipeline.png"

C_MEAS = ("#dbeafe", "#5b8def")   # cheap measurement (blue)
C_GATE = ("#e6f6ef", "#3fae8a")   # gated probe / search (green)
C_TRAIN = ("#fde7e9", "#e8849b")  # the expensive stage (pink)
C_DEF = ("#f4f6f9", "#9aa7b8")    # abstain / default path (grey)
C_PLAIN = ("#ffffff", "#444444")
CONT_FC, CONT_EC = "#eef3fa", "#8fa3bd"

fig, ax = plt.subplots(figsize=(13.5, 21.5))
ax.set_xlim(0, 18.4)
ax.set_ylim(3.8, 32.4)
ax.axis("off")


def box(cx, cy, w, h, text, colors=C_PLAIN, dashed=False, fs=9.5, bold=False):
    fc, ec = colors
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.05,rounding_size=0.14",
                 facecolor=fc, edgecolor=ec, linewidth=1.5,
                 linestyle=(0, (5, 3)) if dashed else "-", zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def diamond(cx, cy, w, h, text, fs=9):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, facecolor="#ffffff", edgecolor="#333333",
                         linewidth=1.5, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4,
            linespacing=1.25)


def container(x0, y0, x1, y1, title):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=CONT_FC,
                           edgecolor=CONT_EC, linewidth=1.8, zorder=1))
    ax.text(x0 + 0.25, y1 - 0.18, title, ha="left", va="top", fontsize=12.5,
            fontweight="bold", color="#3c5170", zorder=2)


def arrow(pts, dashed=False, label=None, lab_dx=0.15, lab_dy=0.0):
    c = "#8a98ab" if dashed else "#333333"
    for i in range(len(pts) - 1):
        last = i == len(pts) - 2
        ax.annotate("", xy=pts[i + 1], xytext=pts[i], zorder=2,
                    arrowprops=dict(arrowstyle="-|>" if last else "-", color=c,
                                    lw=1.6, linestyle=(0, (5, 3)) if dashed else "-"))
    if label:
        mx, my = pts[0]; ex, ey = pts[1]
        ax.text((mx + ex) / 2 + lab_dx, (my + ey) / 2 + lab_dy, label, fontsize=9,
                color=c, ha="left", va="center", zorder=4, style="italic")


# ---- top ---------------------------------------------------------------------
box(9, 31.6, 4.0, 0.9, "unseen dataset\nmetadata + clock", bold=True)
box(9, 30.3, 7.0, 1.0, "1  MEASURE\nfingerprint: neighbouring-pixel correlation\non the raw inputs (CPU seconds, no training)", C_MEAS, fs=9)
box(3.0, 30.3, 4.4, 1.5, "tight budget?\nunder 35 min: skip the searches\n(encoded -> CNN default);\nunder 15 min: also skip the probe", C_DEF, dashed=True, fs=8.5)
arrow([(9, 31.15), (9, 30.85)])

# ---- 2 FAMILY SELECTION ------------------------------------------------------
container(0.8, 23.7, 17.6, 29.4, "2  FAMILY SELECTION      family in {CNN, 1-D TCN, 2-D TCN}")
diamond(9, 28.0, 4.4, 1.5, "encoded data?\n(corr1 < 0.2)")
box(3.7, 26.0, 4.2, 1.5, "image-like:\nfamily := 2-D TCN directly\n(channel-mixing stem if >= 8 ch)\nsaved time goes to training", C_DEF, dashed=True, fs=8.8)
box(13.4, 26.3, 5.2, 1.1, "FAMILY GATE\nshort-train every family\nat matched wall-clock", C_GATE)
diamond(13.4, 24.95, 4.0, 1.25, "leader clearly beats\nthe CNN?\n(margin > noise)", fs=8.4)
box(9.6, 24.25, 3.1, 0.75, "family := leader")
box(16.2, 24.2, 2.9, 0.95, "family := CNN\n(encoded default)", C_DEF, dashed=True, fs=9)
arrow([(9, 29.8), (9, 28.75)])
arrow([(6.8, 28.0), (3.7, 26.78)], dashed=True, label="no: image-like", lab_dx=-2.7, lab_dy=0.45)
arrow([(11.2, 28.0), (13.4, 26.87)], label="yes: encoded", lab_dx=0.3, lab_dy=0.4)
arrow([(13.4, 25.75), (13.4, 25.57)])
arrow([(11.4, 24.95), (9.6, 24.62)], label="yes", lab_dx=-0.75, lab_dy=0.35)
arrow([(15.4, 24.95), (16.2, 24.7)], dashed=True, label="no", lab_dx=-0.15, lab_dy=0.4)

# ---- 3 AUGMENTATION PROBE ----------------------------------------------------
container(0.8, 19.0, 17.6, 23.1, "3  AUGMENTATION PROBE      after the family choice, matched to it")
box(9, 22.0, 6.6, 1.0, "two small nets OF THE CHOSEN FAMILY,\ntrained briefly: with vs without augmentation", C_MEAS, fs=9)
diamond(9, 20.55, 5.0, 1.4, "does OFF win by a clear margin?\n(augmentation is regularisation,\nso OFF leads early)", fs=8.3)
box(5.0, 19.55, 3.0, 0.7, "aug := off + mixup")
box(13.2, 19.55, 3.2, 0.7, "aug := on (the default)", C_DEF, dashed=True, fs=8.7)
arrow([(3.7, 25.25), (3.7, 23.6), (5.9, 22.35)], dashed=True)
arrow([(9.6, 23.87), (9.2, 22.5)])
arrow([(16.2, 23.72), (16.2, 23.3), (12.1, 22.4)], dashed=True)
arrow([(9, 21.5), (9, 21.25)])
arrow([(6.5, 20.55), (5.0, 19.9)], label="yes", lab_dx=-0.7, lab_dy=0.3)
arrow([(11.5, 20.55), (13.2, 19.9)], dashed=True, label="no", lab_dx=0.2, lab_dy=0.35)

# ---- 4 STRUCTURAL SEARCH (depth AND width) -----------------------------------
container(0.8, 13.7, 17.6, 18.6, "4  STRUCTURAL SEARCH      depth in {4, 6, 8} and width in {32, 48, 64}, default-incumbent")
box(9, 17.4, 7.4, 1.2, "short-train depth and width candidates against\nthe INCUMBENT default (D=6, W=48)", C_GATE, fs=9)
diamond(9, 15.7, 4.8, 1.45, "candidate beats the incumbent\nbeyond noise AND survives\nre-training verification?", fs=8.4)
box(5.0, 14.55, 3.3, 0.7, "adopt searched depth / width")
box(13.2, 14.55, 3.0, 0.7, "keep default D=6, W=48", C_DEF, dashed=True, fs=8.8)
arrow([(5.0, 19.2), (5.0, 18.9), (7.4, 18.05)])
arrow([(13.2, 19.2), (13.2, 18.9), (10.7, 18.05)], dashed=True)
arrow([(9, 16.8), (9, 16.45)])
arrow([(6.6, 15.7), (5.0, 14.9)], label="yes", lab_dx=-0.7, lab_dy=0.3)
arrow([(11.4, 15.7), (13.2, 14.9)], dashed=True, label="no", lab_dx=0.2, lab_dy=0.35)

# ---- 5 OPERATION EXPLORATION -------------------------------------------------
container(0.8, 7.7, 17.6, 13.3, "5  OPERATION EXPLORATION      sample architectures over a broad operation vocabulary")
diamond(9, 12.0, 4.6, 1.4, "encoded AND\nbudget remains?")
box(13.4, 10.6, 4.8, 1.0, "sample random architectures,\ntrain each briefly", C_GATE)
diamond(13.4, 9.25, 4.0, 1.25, "clearly better than the\nincumbent AND verified?", fs=8.4)
box(9.5, 8.5, 3.3, 0.75, "adopt sampled architecture")
box(16.2, 8.45, 2.7, 0.95, "discard, keep\nincumbent", C_DEF, dashed=True, fs=9)
arrow([(5.0, 14.2), (5.0, 13.6), (7.2, 12.45)])
arrow([(13.2, 14.2), (13.2, 13.6), (10.9, 12.5)], dashed=True)
arrow([(11.3, 12.0), (13.4, 11.12)], label="yes", lab_dx=0.25, lab_dy=0.4)
arrow([(6.7, 12.0), (1.9, 11.0), (1.9, 6.2), (6.1, 5.85)], dashed=True, label="no", lab_dx=-1.1, lab_dy=0.4)
arrow([(13.4, 10.1), (13.4, 9.88)])
arrow([(11.4, 9.25), (9.5, 8.88)], label="yes", lab_dx=-0.85, lab_dy=0.35)
arrow([(15.4, 9.25), (16.2, 8.95)], dashed=True, label="no", lab_dx=-0.15, lab_dy=0.4)

# ---- 6 TRAIN TO BUDGET -------------------------------------------------------
box(9, 5.7, 6.4, 1.0, "6  TRAIN TO BUDGET\ncosine learning rate annealed to zero, keep the best checkpoint", C_TRAIN, bold=True, fs=9)
box(9, 4.45, 2.8, 0.6, "predictions", bold=True)
arrow([(9.5, 8.12), (9.2, 6.25)])
arrow([(16.2, 7.95), (16.2, 7.0), (11.95, 5.9)], dashed=True)
arrow([(9, 5.18), (9, 4.78)])

# ---- legend ------------------------------------------------------------------
lx, ly = 14.6, 31.95
for dy, (colors, lab, dash) in enumerate([
        (C_MEAS, "cheap measurement", False),
        (C_GATE, "gated probe / search", False),
        (C_TRAIN, "the expensive stage", False),
        (C_DEF, "abstain / default path", True)]):
    box(lx + 0.4, ly - dy * 0.58, 0.7, 0.38, "", colors, dashed=dash)
    ax.text(lx + 0.95, ly - dy * 0.58, lab, fontsize=9.5, va="center")

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"wrote {OUT}")
