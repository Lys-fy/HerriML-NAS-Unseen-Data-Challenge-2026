import random
import numpy as np

def get_rng(seed: int | None = None) -> np.random.Generator:
    """
    Initialize and return a NumPy Generator.
    
    If a seed is provided, the generator is initialized with that seed 
    to ensure deterministic results. Python's built-in random module 
    is also seeded with the same value.

    Args:
        seed: An optional integer to seed the random number generators.

    Returns:
        np.random.Generator: A configured NumPy random generator.
    """
    if seed is not None:
        random.seed(seed)
        return np.random.default_rng(seed)
    return np.random.default_rng()
