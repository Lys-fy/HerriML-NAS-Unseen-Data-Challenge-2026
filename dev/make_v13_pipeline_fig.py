"""Render the v13 decision pipeline as a proposal-style figure (matplotlib).

Boxes = stages, diamonds = runtime decisions, dashed edges = abstain-to-default
and low-budget-collapse routes. Three swimlanes: DataProcessor / NAS / Trainer.
Output: docs/proposal/v13_pipeline.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

# palette
LANE = {"data": "#eaf2fb", "nas": "#eafbf0", "train": "#fbf3ea"}
BOX = "#ffffff"; BOXEDGE = "#3a4a5a"; DIA = "#fff3cd"; DIAEDGE = "#b8860b"
TERM = "#dbe4ee"; TXT = "#1b2733"

fig, ax = plt.subplots(figsize=(8.2, 11.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis("off")


def lane(y0, y1, color, label):
    ax.add_patch(Rectangle((0.2, y0), 9.6, y1 - y0, facecolor=color,
                           edgecolor="none", zorder=0))
    ax.text(0.35, y1 - 0.28, label, fontsize=10.5, style="italic",
            color="#55636f", ha="left", va="top", zorder=1)


def box(x, y, w, h, text, fc=BOX, ec=BOXEDGE, fs=10):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TXT, zorder=4)


def diamond(x, y, w, h, text, fs=9.5):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2),
                          (x - w / 2, y)], closed=True, facecolor=DIA,
                         edgecolor=DIAEDGE, linewidth=1.4, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TXT, zorder=4)


def term(x, y, w, h, text, fs=10):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.45",
                 facecolor=TERM, edgecolor=BOXEDGE, linewidth=1.4, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TXT, zorder=4)


def arrow(p1, p2, dashed=False, label="", lx=0.0, ly=0.0, color=BOXEDGE):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                 linewidth=1.3, color=color, zorder=2,
                 linestyle="--" if dashed else "-",
                 connectionstyle="arc3,rad=0"))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly
        ax.text(mx, my, label, fontsize=8.6, color="#55636f", ha="center",
                va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


# swimlanes
lane(12.55, 14.35, LANE["data"], "DataProcessor")
lane(6.05, 12.5, LANE["nas"], "NAS  ·  measure-gate-train")
lane(0.55, 6.0, LANE["train"], "Trainer  ·  budget-adaptive controller")

# --- terminal + DataProcessor ---
term(5, 14.55, 4.6, 0.62, "unseen dataset  +  clock budget")
box(5, 13.35, 4.6, 0.72, "normalise  +  fingerprint (corr1)")

# --- NAS ---
diamond(5, 12.0, 2.5, 1.0, "encoded?\ncorr1 < 0.2")
box(2.1, 10.7, 2.3, 0.66, "backbone = tcn2d")               # image-like
diamond(7.4, 10.75, 2.4, 1.0, "backbone gate\nconv / tcn1d /\ntcn2d / token", fs=8.6)
box(9.1, 9.35, 1.7, 0.66, "abstain\n-> conv", fs=8.6)
box(5.0, 8.55, 5.2, 0.66, "aug probe  ->  depth probe  ->  attention probe")
box(5.0, 7.25, 5.0, 0.66, "final architecture  +  capacity twin")

# --- Trainer ---
box(5.0, 5.35, 5.4, 0.7, "calibrate step time  ->  plan K seeds x schedule")
box(5.0, 3.95, 5.6, 0.86,
    "train seeds:  EMA · TrivialAugment · val-gated\nsaturation stop releases slice on plateau")
diamond(8.55, 3.95, 1.8, 0.95, "surplus?\nroster < 8", fs=8.4)
box(5.0, 2.5, 5.2, 0.66, "val-gated selection  (ensembles x flip-TTA)")
term(5.0, 1.25, 3.0, 0.62, "predictions")

# collapse box (spans NAS->Trainer via dashed)
box(1.55, 8.55, 1.9, 0.7, "collapse:\ndefault backbone", fc="#f6eefe", ec="#8a5cc0", fs=8.6)

# --- edges ---
arrow((5, 14.24), (5, 13.72))
arrow((5, 12.99), (5, 12.52))
arrow((4.1, 11.7), (2.8, 11.05), label="image-like", ly=0.18)
arrow((6.0, 11.75), (7.15, 11.28), label="encoded", ly=0.18)
arrow((2.1, 10.37), (4.2, 8.9))                             # tcn2d -> probes
arrow((7.4, 10.25), (6.2, 8.9))                             # gate -> probes
arrow((8.25, 10.3), (9.1, 9.7), dashed=True)               # gate -> abstain
arrow((9.1, 9.0), (7.0, 8.9), dashed=True)                 # abstain -> probes
arrow((5, 8.22), (5, 7.6))
arrow((5, 6.92), (5, 5.72))                                 # final -> plan (crosses lane)
arrow((5, 5.0), (5, 4.4))
arrow((5, 3.5), (5, 2.85))
arrow((5, 2.16), (5, 1.58))
# top-up self-loop
arrow((6.2, 4.2), (7.75, 4.0), label="")
arrow((8.55, 3.47), (6.2, 3.7), dashed=True, label="top-up", ly=-0.28)
# collapse route
arrow((3.9, 12.0), (1.55, 8.92), dashed=True, label="budget < 35 min", lx=-0.1, ly=1.3, color="#8a5cc0")
arrow((1.55, 8.19), (3.0, 5.55), dashed=True, color="#8a5cc0")

ax.text(5, 0.28, "Diamonds = runtime decisions.   Dashed = abstain-to-default / low-budget collapse.",
        ha="center", va="center", fontsize=8.8, style="italic", color="#55636f")

plt.tight_layout()
out = "docs/proposal/v13_pipeline.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
