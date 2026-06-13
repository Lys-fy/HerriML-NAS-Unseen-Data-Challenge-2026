import cv2
import numpy as np
from shared.colors import random_rgb, get_distinct_colors

def draw_glossy_stone(
    canvas: np.ndarray, 
    cx: int, 
    cy: int, 
    radius: int, 
    base_color: tuple[int, int, int]
) -> None:
    """
    Draw a premium 3D-looking glossy stone at (cx, cy).
    """
    # 1. Base circle
    cv2.circle(canvas, (cx, cy), radius, base_color, thickness=-1)
    
    # 2. Dark outer rim
    dark_color = (
        max(0, int(base_color[0] * 0.7)),
        max(0, int(base_color[1] * 0.7)),
        max(0, int(base_color[2] * 0.7))
    )
    cv2.circle(canvas, (cx, cy), radius, dark_color, thickness=2)
    
    # 3. Light inner ridge
    light_color = (
        min(255, int(base_color[0] * 1.3)),
        min(255, int(base_color[1] * 1.3)),
        min(255, int(base_color[2] * 1.3))
    )
    cv2.circle(canvas, (cx, cy), int(radius * 0.75), light_color, thickness=2)
    
    # 4. Highlight dot
    hx = cx - int(radius * 0.3)
    hy = cy - int(radius * 0.3)
    hr = max(1, int(radius * 0.15))
    cv2.circle(canvas, (hx, hy), hr, (255, 255, 255), thickness=-1)

def draw_empty_slot(
    canvas: np.ndarray, 
    cx: int, 
    cy: int, 
    radius: int, 
    board_color: tuple[int, int, int],
    empty_color: tuple[int, int, int]
) -> None:
    """
    Draw an empty cutout slot on the board.
    """
    cv2.circle(canvas, (cx, cy), radius, empty_color, thickness=-1)
    
    # Inner shadow (darker board color)
    shadow_color = (
        max(0, int(board_color[0] * 0.4)),
        max(0, int(board_color[1] * 0.4)),
        max(0, int(board_color[2] * 0.4))
    )
    cv2.circle(canvas, (cx, cy), radius, shadow_color, thickness=2)

def render_board(
    board: np.ndarray,
    image_size: int,
    color_mode: str,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Render a Connect Four board state as an RGB image.

    Args:
        board: Board state of shape (H, W, 2).
        image_size: Target image size.
        color_mode: Color mode ('fixed', 'per_sample', 'random_all').
        rng: A NumPy random Generator.

    Returns:
        np.ndarray: Rendered RGB image.
    """
    h, w, _ = board.shape
    
    # Tightly pack the cells so they touch
    cell_size = max(4, min(image_size // w, image_size // h))
    board_w = w * cell_size
    board_h = h * cell_size
    
    ox = (image_size - board_w) // 2
    oy = (image_size - board_h) // 2
    
    # Define color palettes
    if color_mode == "fixed":
        bg_color = (20, 20, 25)       # Dark charcoal background
        board_color = (15, 75, 185)   # Royal blue board
        empty_color = (10, 10, 12)    # Dark holes
        p0_color = (220, 45, 45)      # Glossy red (inactive player)
        p1_color = (245, 215, 20)     # Glossy yellow (active player)
    elif color_mode == "per_sample":
        palette = get_distinct_colors(5, rng, saturation=0.85, value=0.9)
        bg_color = (15, 15, 15)
        board_color = palette[0]
        empty_color = (5, 5, 8)
        p0_color = palette[1]
        p1_color = palette[2]
    else:  # random_all
        bg_color = random_rgb(rng)
        board_color = random_rgb(rng)
        empty_color = random_rgb(rng)
        p0_color = None
        p1_color = None

    canvas = np.full((image_size, image_size, 3), bg_color, dtype=np.uint8)
    
    # Draw board plate rectangle
    cv2.rectangle(
        canvas, 
        (ox, oy), 
        (ox + board_w, oy + board_h), 
        board_color, 
        thickness=-1
    )
    
    # Draw slots (with 0.47 radius to leave almost no gap)
    radius = int(cell_size * 0.47)
    
    for r in range(h):
        for c in range(w):
            cx = ox + c * cell_size + cell_size // 2
            cy = oy + (h - 1 - r) * cell_size + cell_size // 2
            
            is_p0 = board[r, c, 0] == 1
            is_p1 = board[r, c, 1] == 1
            
            if is_p0:
                color = p0_color if p0_color is not None else random_rgb(rng)
                draw_glossy_stone(canvas, cx, cy, radius, color)
            elif is_p1:
                color = p1_color if p1_color is not None else random_rgb(rng)
                draw_glossy_stone(canvas, cx, cy, radius, color)
            else:
                color = empty_color if color_mode != "random_all" else random_rgb(rng)
                draw_empty_slot(canvas, cx, cy, radius, board_color, color)
                
    # Add a premium border around the entire board plate
    border_dark = (
        max(0, int(board_color[0] * 0.5)),
        max(0, int(board_color[1] * 0.5)),
        max(0, int(board_color[2] * 0.5))
    )
    cv2.rectangle(
        canvas, 
        (ox, oy), 
        (ox + board_w, oy + board_h), 
        border_dark, 
        thickness=2
    )
    
    return canvas
