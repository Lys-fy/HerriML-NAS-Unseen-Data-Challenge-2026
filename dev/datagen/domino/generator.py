import numpy as np
from shared.random import get_rng
from shared.noise import apply_image_noise
from shared.io import save_dataset
from domino.utils import (
    generate_domino_chain,
    apply_label_noise,
)
from domino.render import render_dominoes


def generate_dataset(
    num_samples: int,
    image_size: int = 64,
    noise_level: float = 0.0,
    label_noise: float = 0.0,
    color_mode: str = "fixed",
    seed: int | None = None,
    output_dir: str = "./dataset",
    export_hdf5: bool = False,
    min_dominoes: int = 4,
    max_dominoes: int = 9,
) -> None:
    """
    Generate the Domino Rule Chains dataset (binary validity classification).

    Each sample has a random number of dominoes in [min_dominoes, max_dominoes].
    The dataset is balanced: 50 % valid chains, 50 % invalid chains.

    Labels:
        0 = valid chain (all adjacent pairs match)
        1 = invalid chain (exactly one adjacent pair mismatches)

    Output images are always (image_size, image_size, 3).

    Args:
        num_samples: Total number of samples to generate.
        image_size: Height and width of the output images in pixels.
        noise_level: Gaussian noise std-dev factor applied to pixel values.
        label_noise: Probability of flipping the validity label.
        color_mode: 'fixed', 'per_sample', or 'random_all'.
        seed: Optional integer seed for reproducibility.
        output_dir: Root directory for dataset output.
        export_hdf5: Whether to also export an HDF5 file per split.
        min_dominoes: Minimum number of domino tiles per chain.
        max_dominoes: Maximum number of domino tiles per chain (inclusive).
    """
    rng = get_rng(seed)

    images = []
    labels = []

    # Balanced shuffle of validity flags
    half = num_samples // 2
    inject_flags: list[bool] = [False] * half + [True] * (num_samples - half)
    rng.shuffle(inject_flags)

    for inject in inject_flags:
        # Sample a random chain length for this sample
        n = int(rng.integers(min_dominoes, max_dominoes + 1))

        # Generate chain in chain order
        chain = generate_domino_chain(n, rng, inject_error=inject)
        true_label = 1 if inject else 0

        # Render on fixed image_size × image_size canvas
        img = render_dominoes(chain, image_size, color_mode, rng)

        # Apply pixel noise
        img = apply_image_noise(img, noise_level, rng)

        # Optionally corrupt the label
        final_label = apply_label_noise(true_label, label_noise, rng)

        images.append(img)
        labels.append({
            "validity": final_label,
            "true_validity": true_label,
            "num_dominoes": n,
        })

    save_dataset(
        images=images,
        labels=labels,
        output_dir=output_dir,
        export_hdf5=export_hdf5,
        rng=rng,
    )
