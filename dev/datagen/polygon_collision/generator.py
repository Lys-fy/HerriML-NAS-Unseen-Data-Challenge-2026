import numpy as np
from shared.random import get_rng
from shared.noise import apply_image_noise
from shared.io import save_dataset
from polygon_collision.utils import (
    generate_polygon_vertices,
    simulate_trajectory,
    apply_label_noise
)
from polygon_collision.render import render_scene

def generate_dataset(
    num_samples: int,
    image_size: int = 64,
    noise_level: float = 0.0,
    label_noise: float = 0.0,
    color_mode: str = "fixed",
    seed: int | None = None,
    output_dir: str = "./dataset",
    export_hdf5: bool = False,
    num_sides: int = 5,
    num_timesteps: int = 4,
    regular: bool = False,
    velocity_set: list[tuple[float, float]] | None = None,
    edges_same_color: bool = True,
    poly_color_fixed: bool = True,
    obj_color_fixed: bool = True,
    obj_color_fixed_within_timesteps: bool = True
) -> None:
    """
    Generate the Polygon Collision Prediction dataset.

    Generates regular or irregular polygons, simulates a trajectory of a moving
    object with wall bouncing, checks analytically if a collision (bounce) 
    occurs at the next timestep, renders the sequence as a horizontal film strip,
    applies noise, and exports the dataset.

    Args:
        num_samples: Number of samples to generate.
        image_size: Height (and width of each sub-frame) of the output.
        noise_level: Standard deviation factor for image noise.
        label_noise: Probability of flipping the collision label.
        color_mode: Color mode ('fixed', 'per_sample', 'random_all').
        seed: Random seed for reproducibility.
        output_dir: Target output directory.
        export_hdf5: Whether to export datasets in HDF5 format.
        num_sides: Number of sides of the polygon.
        num_timesteps: Number of rendered timesteps.
        regular: True for regular polygon, False for irregular.
        velocity_set: List of possible (vx, vy) velocities.
        edges_same_color: True if all polygon edges have the same color.
        poly_color_fixed: True if polygon colors are fixed across samples.
        obj_color_fixed: True if object color is fixed across samples.
        obj_color_fixed_within_timesteps: True if object color is fixed within timesteps.
    """
    rng = get_rng(seed)
    
    if velocity_set is None:
        # Default velocity set (step size of ~4-6 pixels in various directions)
        velocity_set = [
            (-5.0, 0.0), (5.0, 0.0), (0.0, -5.0), (0.0, 5.0),
            (-4.0, -4.0), (-4.0, 4.0), (4.0, -4.0), (4.0, 4.0),
            (-3.0, -2.0), (3.0, 2.0), (-2.0, 3.0), (2.0, -3.0)
        ]
        
    images = []
    labels = []
    
    collision_pool = []
    no_collision_pool = []
    
    half_needed = num_samples // 2
    attempts = 0
    max_attempts = num_samples * 50
    
    while (len(collision_pool) < half_needed or len(no_collision_pool) < half_needed) and attempts < max_attempts:
        attempts += 1
        
        # 1. Generate polygon vertices
        vertices = generate_polygon_vertices(num_sides, image_size, regular, rng)
        
        # 2. Simulate trajectory (returns path, list of velocities at each step, and label)
        sim = simulate_trajectory(vertices, num_timesteps, velocity_set, image_size, rng)
        if sim is None:
            continue
            
        path, velocities, true_label = sim
        initial_velocity = velocities[0]
        
        if true_label == 1:
            if len(collision_pool) < num_samples:
                collision_pool.append((vertices, path, initial_velocity, true_label))
        else:
            if len(no_collision_pool) < num_samples:
                no_collision_pool.append((vertices, path, initial_velocity, true_label))
                
    # Combine pools to select final set of samples
    selected_samples = []
    if len(collision_pool) >= half_needed and len(no_collision_pool) >= half_needed:
        rng.shuffle(collision_pool)
        rng.shuffle(no_collision_pool)
        selected_samples.extend(collision_pool[:half_needed])
        selected_samples.extend(no_collision_pool[:(num_samples - half_needed)])
    else:
        combined = collision_pool + no_collision_pool
        rng.shuffle(combined)
        selected_samples = combined[:num_samples]
        
    rng.shuffle(selected_samples)
    
    # Process selected samples (render, noise, label-noise, saving)
    for vertices, path, init_vel, true_label in selected_samples:
        img = render_scene(
            poly_vertices=vertices,
            path=path,
            num_timesteps=num_timesteps,
            image_size=image_size,
            color_mode=color_mode,
            rng=rng,
            edges_same_color=edges_same_color,
            poly_color_fixed=poly_color_fixed,
            obj_color_fixed=obj_color_fixed,
            obj_color_fixed_within_timesteps=obj_color_fixed_within_timesteps
        )
        
        img_noisy = apply_image_noise(img, noise_level, rng)
        final_label = apply_label_noise(true_label, label_noise, rng)
        
        images.append(img_noisy)
        labels.append({
            "collision": final_label,
            "true_collision": true_label,
            "velocity_x": init_vel[0],
            "velocity_y": init_vel[1]
        })
        
    save_dataset(
        images=images,
        labels=labels,
        output_dir=output_dir,
        export_hdf5=export_hdf5,
        rng=rng
    )
