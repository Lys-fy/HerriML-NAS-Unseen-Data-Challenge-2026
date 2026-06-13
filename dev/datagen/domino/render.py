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

    pip_r = max(3, min(hw, hh) // 9)
    for dx, dy in _PIP_LAYOUTS.get(val, []):
        px = int(cx + dx * hw * 1.35)
        py = int(cy + dy * hh * 1.35)
        cv2.circle(canvas, (px, py), pip_r, pip_color, thickness=-1)


def render_dominoes(
    dominoes: list[Domino],
    image_size: int,
    color_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Render domino chain on a fixed image_size × image_size canvas.

    Dominoes are placed exactly on the walk grid. Each domino spans two
    adjacent grid cells and is split into two clearly separated halves, so the
    matching rule is visible and the pip count remains readable.

    Args:
        dominoes: Chain-ordered list of Domino objects.
        image_size: Output image height and width in pixels.
        color_mode: 'fixed', 'per_sample', or 'random_all'.
        rng: NumPy random Generator.

    Returns:
        np.ndarray of shape (image_size, image_size, 3), dtype uint8.
    """
    n = len(dominoes)
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # ── Layout: map walk grid → pixel space ──────────────────────────────────
    # A horizontal domino occupies (row, col) and (row, col + 1).
    # A vertical domino occupies (row, col) and (row + 1, col).
    rows = [d.row + (1 if d.orientation == 'V' else 0) for d in dominoes]
    cols = [d.col + (1 if d.orientation == 'H' else 0) for d in dominoes]
    min_row = min(d.row for d in dominoes)
    min_col = min(d.col for d in dominoes)
    max_row = max(rows)
    max_col = max(cols)

    grid_rows = max_row - min_row + 1
    grid_cols = max_col - min_col + 1
    margin = max(4, image_size // 32)
    usable_size = max(1, image_size - 2 * margin)
    cell_size = max(12, usable_size // max(grid_rows, grid_cols))

    offset_x = margin + (usable_size - grid_cols * cell_size) // 2
    offset_y = margin + (usable_size - grid_rows * cell_size) // 2

    # ── Background ────────────────────────────────────────────────────────────
    if color_mode == "fixed":
        bg_color: tuple[int, int, int] = (30, 95, 40)
    elif color_mode == "per_sample":
        bg_color = get_distinct_colors(1, rng, saturation=0.6, value=0.35)[0]
    else:
        bg_color = random_rgb(rng)
    canvas[:] = bg_color

    # ── Colour palette ────────────────────────────────────────────────────────
    if color_mode == "fixed":
        face_color_base: tuple[int, int, int] | None = (248, 244, 228)
        pip_color_base: tuple[int, int, int] | None  = (20, 20, 20)
        border_color_base: tuple[int, int, int] | None = (55, 55, 55)
    elif color_mode == "per_sample":
        palette = get_distinct_colors(3, rng, saturation=0.7, value=0.90)
        face_color_base = palette[0]
        pip_color_base  = palette[1]
        border_color_base = (30, 30, 30)
    else:  # random_all — pick per domino below
        face_color_base = None
        pip_color_base  = None
        border_color_base = None

    # ── Draw each domino in chain order ───────────────────────────────────────
    tile_gap = max(2, cell_size // 18)
    half_gap = max(1, cell_size // 28)

    for d in dominoes:
        grid_col = d.col - min_col
        grid_row = d.row - min_row
        x0 = offset_x + grid_col * cell_size + tile_gap
        y0 = offset_y + grid_row * cell_size + tile_gap

        face_c  = face_color_base  if face_color_base  is not None else random_rgb(rng)
        pip_c   = pip_color_base   if pip_color_base   is not None else random_rgb(rng)
        bord_c  = border_color_base if border_color_base is not None else random_rgb(rng)

        if d.orientation == 'H':
            # Full domino spans two grid cells horizontally.
            x1 = x0 + 2 * cell_size - 2 * tile_gap
            y1 = y0 + cell_size - 2 * tile_gap
            divider_x = x0 + cell_size - tile_gap

            left_cx = (x0 + divider_x - half_gap) // 2
            right_cx = (divider_x + half_gap + x1) // 2
            cy = (y0 + y1) // 2
            half_tile_hw = max(4, (divider_x - half_gap - x0) // 2)
            half_tile_hh = max(4, (y1 - y0) // 2)

            _draw_half(canvas, left_cx, cy,
                       half_tile_hw, half_tile_hh,
                       d.val1, face_c, pip_c, bord_c)
            _draw_half(canvas, right_cx, cy,
                       half_tile_hw, half_tile_hh,
                       d.val2, face_c, pip_c, bord_c)

            dw = max(2, cell_size // 18)
            cv2.line(canvas,
                     (divider_x, y0),
                     (divider_x, y1),
                     bord_c, thickness=dw)

        else:  # 'V'
            # Full domino spans two grid cells vertically.
            x1 = x0 + cell_size - 2 * tile_gap
            y1 = y0 + 2 * cell_size - 2 * tile_gap
            divider_y = y0 + cell_size - tile_gap

            cx = (x0 + x1) // 2
            top_cy = (y0 + divider_y - half_gap) // 2
            bottom_cy = (divider_y + half_gap + y1) // 2
            half_tile_hw = max(4, (x1 - x0) // 2)
            half_tile_hh = max(4, (divider_y - half_gap - y0) // 2)

            _draw_half(canvas, cx, top_cy,
                       half_tile_hw, half_tile_hh,
                       d.val1, face_c, pip_c, bord_c)
            _draw_half(canvas, cx, bottom_cy,
                       half_tile_hw, half_tile_hh,
                       d.val2, face_c, pip_c, bord_c)

            dw = max(2, cell_size // 18)
            cv2.line(canvas,
                     (x0, divider_y),
                     (x1, divider_y),
                     bord_c, thickness=dw)

    return canvas
