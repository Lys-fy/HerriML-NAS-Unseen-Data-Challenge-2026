import os
import csv
import numpy as np
from PIL import Image

def save_dataset(
    images: list[np.ndarray],
    labels: list[dict],
    output_dir: str,
    export_hdf5: bool = False,
    rng: np.random.Generator | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> None:
    """
    Save the dataset to the specified directory with train/val/test splits.
    
    Always exports PNG images and a labels.csv file for each split.
    If export_hdf5 is True, also creates a .h5 file for each split containing 
    the stacked image array and individual datasets for each label field.

    Args:
        images: List of RGB images as NumPy arrays (uint8).
        labels: List of dictionaries containing the label names and values.
        output_dir: Root directory for the generated dataset.
        export_hdf5: Whether to also export the dataset in HDF5 format.
        rng: A NumPy random Generator used for shuffling.
        train_ratio: Proportion of samples in the training set.
        val_ratio: Proportion of samples in the validation set.
        test_ratio: Proportion of samples in the test set.
    """
    total = len(images)
    if total == 0:
        return

    # Shuffle indices deterministically using the provided rng
    indices = np.arange(total)
    if rng is not None:
        rng.shuffle(indices)

    # Calculate split bounds
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:]
    }

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_idx in splits.items():
        if len(split_idx) == 0:
            continue

        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_images = [images[i] for i in split_idx]
        split_labels = [labels[i] for i in split_idx]

        # 1. Export as PNG and CSV
        images_dir = os.path.join(split_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        csv_path = os.path.join(split_dir, "labels.csv")
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            if len(split_labels) > 0:
                fieldnames = ["filename"] + list(split_labels[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for idx, (img, lbl) in enumerate(zip(split_images, split_labels)):
                    filename = f"images/{idx:06d}.png"
                    img_path = os.path.join(split_dir, filename)

                    # Save PNG image
                    pil_img = Image.fromarray(img)
                    pil_img.save(img_path)

                    row = {"filename": filename}
                    row.update(lbl)
                    writer.writerow(row)

        # 2. Export as HDF5 if requested
        if export_hdf5:
            try:
                import h5py
                h5_path = os.path.join(output_dir, f"{split_name}.h5")
                with h5py.File(h5_path, "w") as h5_file:
                    # Stack images into a single array (N, H, W, 3)
                    img_array = np.stack(split_images, axis=0)
                    h5_file.create_dataset("images", data=img_array, compression="gzip")

                    # Create a dataset for each label key
                    for key in split_labels[0].keys():
                        vals = [lbl[key] for lbl in split_labels]
                        
                        # Handle strings vs numbers
                        first_val = vals[0]
                        if isinstance(first_val, (str, bytes)):
                            dt = h5py.special_dtype(vlen=str)
                            vals_arr = np.array(vals, dtype=dt)
                        elif isinstance(first_val, (list, tuple)):
                            # Convert tuples/lists to numpy arrays
                            vals_arr = np.array(vals)
                        else:
                            vals_arr = np.array(vals)
                            
                        h5_file.create_dataset(key, data=vals_arr)
            except ImportError:
                # Fallback if h5py is not installed
                import warnings
                warnings.warn("h5py is not installed. Skipping HDF5 export.")
