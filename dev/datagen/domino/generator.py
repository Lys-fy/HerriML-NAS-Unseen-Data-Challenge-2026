import numpy as np
from shared.random import get_rng
from shared.noise import apply_image_noise
from shared.io import save_dataset
from domino.utils import (
    generate_domino_chain,
    apply_label_noise,
    apply_label_noise_multiclass,
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
    task: str = "localisation",
    domino_count: int = 5,
    tile: int = 34,
) -> None:
    """
    Generate the Domino Rule Chains dataset.

    The chain is drawn left to right (vendored rewrite 2026-06-13), large
    enough that the pips are countable; consecutive dominoes are separated by a
    gap (the junction the chain rule checks).

    Two tasks:
      - "localisation" (default): a fixed `domino_count`-long chain. The label is
        the junction index of the single mismatch, or a dedicated "valid" class.
        Classes: 0 = valid, k = mismatch at junction (k-1) for k in 1..L-1, so
        `domino_count` classes total, balanced across all of them.
      - "validity": variable length [min_dominoes, max_dominoes], binary label
        (0 = valid, 1 = one mismatch). The original task framing.

    All images share one shape (the canvas is sized for the max chain length).
    The canonical training label is written to the "label" column.

    Args:
        num_samples: Total number of samples to generate.
        image_size: Kept for CLI compatibility; tile size is set by `tile`.
        noise_level: Gaussian pixel-noise std-dev factor.
        label_noise: Probability of corrupting the label (multiclass for the
            localisation task, binary flip for validity).
        color_mode: 'fixed', 'per_sample', or 'random_all'.
        seed: Optional integer seed.
        output_dir: Root output directory.
        export_hdf5: Whether to also export an HDF5 file per split.
        min_dominoes / max_dominoes: chain-length range for the validity task.
        task: "localisation" or "validity".
        domino_count: fixed chain length for the localisation task.
        tile: half-cell side in pixels (pip readability).
    """
    rng = get_rng(seed)
    images: list = []
    labels: list = []

    if task == "localisation":
        L = domino_count
        n_classes = L  # 0 = valid; 1..L-1 = mismatch at junction 0..L-2
        # Even per-class targets, remainder spread over the first classes.
        base = num_samples // n_classes
        counts = [base + (1 if i < num_samples - base * n_classes else 0)
                  for i in range(n_classes)]
        specs = [cls for cls in range(n_classes) for _ in range(counts[cls])]
        rng.shuffle(specs)

        for cls in specs:
            if cls == 0:
                chain, boundary = generate_domino_chain(L, rng, inject_error=False)
            else:
                chain, boundary = generate_domino_chain(
                    L, rng, inject_error=True, error_boundary=cls - 1)
            true_label = 0 if boundary < 0 else boundary + 1
            img = render_dominoes(chain, image_size, color_mode, rng,
                                  max_count=L, tile=tile)
            img = apply_image_noise(img, noise_level, rng)
            final_label = apply_label_noise_multiclass(
                true_label, n_classes, label_noise, rng)
            images.append(img)
            labels.append({
                "label": final_label,
                "true_label": true_label,
                "mismatch_junction": boundary,  # -1 if valid
                "num_dominoes": L,
            })

    elif task == "validity":
        half = num_samples // 2
        inject_flags = [False] * half + [True] * (num_samples - half)
        rng.shuffle(inject_flags)
        for inject in inject_flags:
            n = int(rng.integers(min_dominoes, max_dominoes + 1))
            chain, boundary = generate_domino_chain(n, rng, inject_error=inject)
            true_label = 1 if inject else 0
            img = render_dominoes(chain, image_size, color_mode, rng,
                                  max_count=max_dominoes, tile=tile)
            img = apply_image_noise(img, noise_level, rng)
            final_label = apply_label_noise(true_label, label_noise, rng)
            images.append(img)
            labels.append({
                "label": final_label,
                "true_label": true_label,
                "mismatch_junction": boundary,
                "num_dominoes": n,
            })
    else:
        raise ValueError(f"unknown domino task '{task}'")

    save_dataset(
        images=images,
        labels=labels,
        output_dir=output_dir,
        export_hdf5=export_hdf5,
        rng=rng,
    )
