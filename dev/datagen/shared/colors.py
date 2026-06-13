import colorsys
import numpy as np

def random_rgb(rng: np.random.Generator) -> tuple[int, int, int]:
    """
    Generate a random RGB color tuple in range [0, 255].

    Args:
        rng: A NumPy random Generator.

    Returns:
        tuple[int, int, int]: A random RGB color.
    """
    return tuple(rng.integers(0, 256, size=3).tolist())

def get_distinct_colors(
    n: int, 
    rng: np.random.Generator,
    saturation: float = 0.8,
    value: float = 0.9
) -> list[tuple[int, int, int]]:
    """
    Generate n distinct RGB colors using evenly spaced hues with a random offset.

    Args:
        n: Number of distinct colors to generate.
        rng: A NumPy random Generator.
        saturation: Saturation value in HSV [0.0, 1.0].
        value: Value/brightness in HSV [0.0, 1.0].

    Returns:
        list[tuple[int, int, int]]: List of RGB tuples in range [0, 255].
    """
    if n <= 0:
        return []
    
    start_hue = rng.uniform(0.0, 1.0)
    hues = [(start_hue + i / n) % 1.0 for i in range(n)]
    
    colors = []
    for h in hues:
        # Add a tiny bit of jitter to saturation and value for realism
        s = clip(saturation + rng.uniform(-0.1, 0.1), 0.2, 1.0)
        v = clip(value + rng.uniform(-0.1, 0.1), 0.2, 1.0)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
        
    return colors

def clip(val: float, min_val: float, max_val: float) -> float:
    """Helper to clip a float value between a minimum and maximum."""
    return max(min_val, min(val, max_val))
