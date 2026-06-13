import numpy as np
from mathvision.utils import evaluate_polynomial

def hsv_to_rgb_vectorized(
    h: np.ndarray, 
    s: np.ndarray, 
    v: np.ndarray
) -> np.ndarray:
    """
    Vectorized conversion from HSV to RGB color space.

    Args:
        h: Hue array of shape (H, W), values in [0.0, 1.0].
        s: Saturation array of shape (H, W), values in [0.0, 1.0].
        v: Value array of shape (H, W), values in [0.0, 1.0].

    Returns:
        np.ndarray: RGB image array of shape (H, W, 3) and type uint8.
    """
    hi = np.floor(h * 6).astype(int) % 6
    f = (h * 6) - np.floor(h * 6)
    p = v * (1.0 - s)
    q = v * (1.0 - (f * s))
    t = v * (1.0 - ((1.0 - f) * s))
    
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    
    mask0 = (hi == 0)
    r[mask0], g[mask0], b[mask0] = v[mask0], t[mask0], p[mask0]
    
    mask1 = (hi == 1)
    r[mask1], g[mask1], b[mask1] = q[mask1], v[mask1], p[mask1]
    
    mask2 = (hi == 2)
    r[mask2], g[mask2], b[mask2] = p[mask2], v[mask2], t[mask2]
    
    mask3 = (hi == 3)
    r[mask3], g[mask3], b[mask3] = p[mask3], q[mask3], v[mask3]
    
    mask4 = (hi == 4)
    r[mask4], g[mask4], b[mask4] = t[mask4], p[mask4], v[mask4]
    
    mask5 = (hi == 5)
    r[mask5], g[mask5], b[mask5] = v[mask5], p[mask5], q[mask5]
    
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)

def render_polynomial(
    coeffs: np.ndarray,
    image_size: int,
    color_mode: str,
    rng: np.random.Generator,
    zoom_and_translate: bool = True
) -> np.ndarray:
    """
    Render a complex polynomial over a grid in the complex plane.

    Encoding:
        - hue = complex phase / argument
        - brightness = magnitude (using periodic log-magnitude lines)

    Args:
        coeffs: Polynomial coefficients.
        image_size: Target image size (width and height).
        color_mode: Color mode ('fixed', 'per_sample', 'random_all').
        rng: A NumPy random Generator.
        zoom_and_translate: Whether to apply random translation/zoom to the viewport.

    Returns:
        np.ndarray: Rendered RGB image of shape (image_size, image_size, 3).
    """
    # Define complex plane viewport
    if zoom_and_translate:
        zoom = rng.uniform(0.5, 3.0)
        dx = rng.uniform(-1.5, 1.5)
        dy = rng.uniform(-1.5, 1.5)
    else:
        zoom = 1.0
        dx = 0.0
        dy = 0.0
        
    x = np.linspace(dx - 2.0 * zoom, dx + 2.0 * zoom, image_size)
    # y coordinates decreasing to match top-to-bottom image coordinates
    y = np.linspace(dy + 2.0 * zoom, dy - 2.0 * zoom, image_size)
    
    X, Y = np.meshgrid(x, y)
    z = X + 1j * Y
    
    w = evaluate_polynomial(coeffs, z)
    
    # 1. Compute Hue (Phase)
    phase = np.angle(w)  # Range [-pi, pi]
    h = (phase + np.pi) / (2.0 * np.pi)  # Normalize to [0.0, 1.0]
    
    # 2. Compute Value (Magnitude-based periodic grid)
    mag = np.abs(w)
    log_mag = np.log2(np.maximum(mag, 1e-9))
    # Make value periodic to create grid lines (showing roots clearly)
    v = 0.4 + 0.6 * (log_mag - np.floor(log_mag))
    
    # 3. Compute Saturation
    s = np.ones_like(h)
    
    # Apply color modes
    if color_mode == "fixed":
        # Standard coloring
        pass
    elif color_mode == "per_sample":
        # Random hue shift and saturation constant per sample
        h_shift = rng.uniform(0.0, 1.0)
        h = (h + h_shift) % 1.0
        s = s * rng.uniform(0.7, 1.0)
    elif color_mode == "random_all":
        # Random hue shift and saturation per pixel
        h_shift = rng.uniform(0.0, 1.0, size=h.shape)
        h = (h + h_shift) % 1.0
        s = s * rng.uniform(0.5, 1.0, size=s.shape)
        v = np.clip(v + rng.normal(0.0, 0.1, size=v.shape), 0.0, 1.0)
        
    return hsv_to_rgb_vectorized(h, s, v)
