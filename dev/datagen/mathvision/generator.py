import numpy as np
from shared.random import get_rng
from shared.noise import apply_image_noise
from shared.io import save_dataset
from mathvision.utils import sample_polynomial_coefficients, apply_label_noise
from mathvision.render import render_polynomial

def generate_dataset(
    num_samples: int,
    image_size: int = 64,
    noise_level: float = 0.0,
    label_noise: float = 0.0,
    color_mode: str = "fixed",
    seed: int | None = None,
    output_dir: str = "./dataset",
    export_hdf5: bool = False,
    max_degree: int = 5
) -> None:
    """
    Generate the MathVision dataset.
    
    Generates complex polynomial phase portraits, adds noise, applies label 
    noise, and saves the output split into train, val, and test subsets.

    Args:
        num_samples: Number of samples to generate.
        image_size: Size of the square RGB images.
        noise_level: Standard deviation factor for image noise.
        label_noise: Probability of replacing the true label with a noisy label.
        color_mode: Color mode ('fixed', 'per_sample', 'random_all').
        seed: Random seed for reproducibility.
        output_dir: Target output directory.
        export_hdf5: Whether to export datasets in HDF5 format.
        max_degree: Maximum degree of the generated polynomials (sampled 1..max_degree).
    """
    rng = get_rng(seed)
    
    images = []
    labels = []
    
    for _ in range(num_samples):
        # Sample the true degree
        true_degree = int(rng.integers(1, max_degree + 1))
        
        # Sample polynomial coefficients
        coeffs = sample_polynomial_coefficients(true_degree, rng)
        
        # Render the polynomial phase portrait
        img = render_polynomial(
            coeffs=coeffs,
            image_size=image_size,
            color_mode=color_mode,
            rng=rng,
            zoom_and_translate=True
        )
        
        # Apply image noise
        img_noisy = apply_image_noise(img, noise_level, rng)
        
        # Apply label noise
        final_label = apply_label_noise(true_degree, max_degree, label_noise, rng)
        
        images.append(img_noisy)
        labels.append({
            "degree": final_label,
            "true_degree": true_degree
        })
        
    # Save the dataset using the shared I/O utility
    save_dataset(
        images=images,
        labels=labels,
        output_dir=output_dir,
        export_hdf5=export_hdf5,
        rng=rng
    )
