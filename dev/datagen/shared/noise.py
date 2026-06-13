import numpy as np

def apply_image_noise(
    image: np.ndarray, 
    noise_level: float, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Apply Gaussian noise to an RGB image.

    Args:
        image: A NumPy array representing the RGB image (uint8).
        noise_level: Level of noise from 0.0 (none) to 1.0 (heavy noise).
        rng: A NumPy random Generator.

    Returns:
        np.ndarray: The noisy image as a uint8 array.
    """
    if noise_level <= 0.0:
        return image
    
    # Standard deviation scales with noise_level
    std = noise_level * 128.0
    noise = rng.normal(0, std, size=image.shape)
    
    noisy = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    return noisy
