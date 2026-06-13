import cv2
import numpy as np
from domino.utils import Domino
from shared.colors import random_rgb, get_distinct_colors


# Standard pip layouts in normalized [-1, 1] space (dx, dy)
_PIP_LAYOUTS: dict[int, list[tuple[float, float]]] = {
    0: [],
    1: [(0.0, 0.0)],
    2: [(-0.32, -0.32), (0.32, 0.32)],
    3: [(-0.32, -0.32), (0.0, 0.0), (0.32, 0.32)],
    4: [(-0.32, -0.32), (-0.32, 0.32), (0.32, -0.32), (0.32, 0.32)],
    5: [(-0.32, -0.32), (-0.32, 0.32), (0.0, 0.0), (0.32, -0.32), (0.32, 0.32)],
    6: [(-0.32, -0.32), (-0.32, 0.0), (-0.32, 0.32),
        (0.32, -0.32), (0.32, 0.0), (0.32, 0.32)],
}


def _draw_half(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    hw: int,
    hh: int,
    val: int,
    face_color: tuple[int, int, int],
    pip_color: tuple[int, int, int],
    border_color: tuple[int, int, int],
) -> None:
    """
    Draw one half of a domino tile — a filled rectangle with border and pips.

    Args:
        canvas: Image to draw on.
        cx, cy: Centre pixel of this half.
        hw, hh: Half-width and half-height of this half-tile cell.
        val: Pip value (0-6).
        face_color: Fill colour.
        pip_color: Pip dot colour.
        border_color: Border line colour.
    """
    x0, y0 = cx - hw, cy - hh
    x1, y1 = cx + hw, cy + hh

    cv2.rectangle(canvas, (x0, y0), (x1, y1), face_color, thickness=-1)

    bw = max(2, min(hw, hh) // 10)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), border_color, thickness=bw)

    pip_r = max(3, min(hw, hh) // 4)
    for dx, dy in _PIP_LAYOUTS.get(val, []):
        px = int(cx + dx * hw * 1.55)
        py = int(cy + dy * hh * 1.55)
        cv2.circle(canvas, (px, py), pip_r, pip_color, thickness=-1)


def render_dominoes(
    dominoes: list[Domino],
    image_size: int,
    color_mode: str,
    rng: np.random.Generator,
    max_count: int | None = None,
    tile: int = 34,
) -> np.ndarray:
    """
    Render a domino chain as a single left-to-right row (vendored rewrite
    2026-06-13).

    Each domino is two adjacent square halves; consecutive dominoes are
    separated by a small gap, so the touching halves that the chain rule checks
    (domino[i].val2 next to domino[i+1].val1) flank that gap and are read at a
    glance. The 2-D self-avoiding-walk layout is dropped: chain order is simply
    left to right, which both removes the "recover the order" sub-problem and
    frees pixels for large, countable pips.

    The canvas is sized for `max_count` dominoes so every image in a run shares
    dimensions; shorter chains are left-aligned with empty background on the
    right. `image_size` is kept for call-site compatibility but the tile size is
    controlled by `tile` (half-cell side in px).

    Returns:
        np.ndarray of shape (H, W, 3), uint8, with
        H = 2*margin + tile and W = 2*margin + max_count*2*tile + (max_count-1)*gap.
    """
    n = len(dominoes)
    if max_count is None:
        max_count = n
    margin = 5
    gap = max(4, tile // 5)          # the junction gap between consecutive dominoes
    half = tile
    domino_w = 2 * half
    H = 2 * margin + half
    W = 2 * margin + max_count * domino_w + (max_count - 1) * gap

    # Background + palette (same colour semantics as the original)
    if color_mode == "fixed":
        bg_color = (30, 95, 40)
        face_color_base = (248, 244, 228)
        pip_color_base = (20, 20, 20)
        border_color_base = (55, 55, 55)
    elif color_mode == "per_sample":
        bg_color = get_distinct_colors(1, rng, saturation=0.6, value=0.35)[0]
        palette = get_distinct_colors(3, rng, saturation=0.7, value=0.90)
        face_color_base, pip_color_base, border_color_base = palette[0], palette[1], (30, 30, 30)
    else:  # random_all
        bg_color = random_rgb(rng)
        face_color_base = pip_color_base = border_color_base = None

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = bg_color

    hw = hh = half // 2
    cy = margin + hh
    x = margin
    for d in dominoes:
        face_c = face_color_base if face_color_base is not None else random_rgb(rng)
        pip_c = pip_color_base if pip_color_base is not None else random_rgb(rng)
        bord_c = border_color_base if border_color_base is not None else random_rgb(rng)

        left_cx = x + hw
        right_cx = x + half + hw
        _draw_half(canvas, left_cx, cy, hw, hh, d.val1, face_c, pip_c, bord_c)
        _draw_half(canvas, right_cx, cy, hw, hh, d.val2, face_c, pip_c, bord_c)

        dw = max(2, half // 16)
        cv2.line(canvas, (x + half, cy - hh), (x + half, cy + hh), bord_c, thickness=dw)
        x += domino_w + gap

    return canvas
