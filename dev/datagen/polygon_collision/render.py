import cv2
import numpy as np
from shared.colors import random_rgb, get_distinct_colors

def render_scene(
    poly_vertices: np.ndarray,
    path: list[tuple[float, float]],
    num_timesteps: int,
    image_size: int,
    color_mode: str,
    rng: np.random.Generator,
    edges_same_color: bool = True,
    poly_color_fixed: bool = True,
    obj_color_fixed: bool = True,
    obj_color_fixed_within_timesteps: bool = True
) -> np.ndarray:
    """
    Render the trajectory as a film strip of side-by-side frames.
    
    The output image has dimensions: image_size (height) x (image_size * num_timesteps) (width).
    Each sub-frame shows the polygon and the object at a single timestep.

    Args:
        poly_vertices: Vertices of the polygon (N, 2).
        path: Trajectory points of length num_timesteps.
        num_timesteps: Number of rendered timesteps/frames.
        image_size: Height (and width of each individual frame).
        color_mode: Color mode ('fixed', 'per_sample', 'random_all').
        rng: A NumPy random Generator.
        edges_same_color: Whether polygon edges have the same color.
        poly_color_fixed: Whether polygon colors are fixed across samples.
        obj_color_fixed: Whether object colors are fixed across samples.
        obj_color_fixed_within_timesteps: Whether object color is constant across timesteps.

    Returns:
        np.ndarray: Rendered RGB image of shape (image_size, image_size * num_timesteps, 3).
    """
    width = image_size * num_timesteps
    
    # 1. Determine background and line colors
    bg_color = (20, 20, 22) if color_mode != "random_all" else random_rgb(rng)
    canvas = np.full((image_size, width, 3), bg_color, dtype=np.uint8)
    
    # 2. Determine polygon edge colors for each frame
    num_edges = len(poly_vertices)
    if color_mode == "random_all":
        poly_colors = [random_rgb(rng) for _ in range(num_edges)]
    elif poly_color_fixed:
        if edges_same_color:
            poly_colors = [(220, 220, 220)] * num_edges
        else:
            poly_colors = [
                (255, 80, 80), (80, 255, 80), (80, 80, 255),
                (255, 255, 80), (255, 80, 255), (80, 255, 255)
            ]
            poly_colors = [poly_colors[i % len(poly_colors)] for i in range(num_edges)]
    else:
        if edges_same_color:
            color = random_rgb(rng)
            poly_colors = [color] * num_edges
        else:
            poly_colors = get_distinct_colors(num_edges, rng, saturation=0.8, value=0.9)

    # 3. Determine object colors for each timestep
    rendered_len = min(num_timesteps, len(path))
    if color_mode == "random_all":
        obj_colors = [random_rgb(rng) for _ in range(rendered_len)]
    elif obj_color_fixed:
        if obj_color_fixed_within_timesteps:
            obj_colors = [(0, 255, 255)] * rendered_len  # Cyan
        else:
            # Color gradient from cyan to magenta
            obj_colors = []
            for t in range(rendered_len):
                ratio = t / max(1, rendered_len - 1)
                r = int(0 * (1 - ratio) + 255 * ratio)
                g = int(255 * (1 - ratio) + 0 * ratio)
                b = int(255 * (1 - ratio) + 255 * ratio)
                obj_colors.append((r, g, b))
    else:
        if obj_color_fixed_within_timesteps:
            color = random_rgb(rng)
            obj_colors = [color] * rendered_len
        else:
            obj_colors = get_distinct_colors(rendered_len, rng, saturation=0.9, value=0.9)
            
    # Object radius
    radius = max(2, int(image_size * 0.04))
    
    # Draw each frame
    for t in range(num_timesteps):
        ox = t * image_size
        
        # Draw polygon edges for this frame
        for i in range(num_edges):
            # Shift x coordinate by frame offset ox
            p1 = (int(poly_vertices[i][0] + ox), int(poly_vertices[i][1]))
            p2 = (int(poly_vertices[(i + 1) % num_edges][0] + ox), int(poly_vertices[(i + 1) % num_edges][1]))
            cv2.line(canvas, p1, p2, poly_colors[i], thickness=2)
            
        # Draw the object only at the current position for this frame
        if t < len(path):
            cx = int(path[t][0] + ox)
            cy = int(path[t][1])
            cv2.circle(canvas, (cx, cy), radius, obj_colors[t], thickness=-1)
            cv2.circle(canvas, (cx, cy), radius, (0, 0, 0), thickness=1)
            
        # Draw thin vertical divider between frames
        if t > 0:
            cv2.line(canvas, (ox, 0), (ox, image_size), (50, 50, 55), thickness=1)
            
    return canvas
