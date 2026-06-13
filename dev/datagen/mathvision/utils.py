import numpy as np

def sample_polynomial_coefficients(
    degree: int, 
    rng: np.random.Generator
) -> np.ndarray:
    """
    Sample complex coefficients for a polynomial of a given degree.
    
    The coefficients a_k are complex numbers with real and imaginary 
    parts sampled uniformly from [-1.0, 1.0]. The leading coefficient 
    a_d is guaranteed to be non-zero (amplitude >= 0.1).

    Args:
        degree: The degree of the polynomial.
        rng: A NumPy random Generator.

    Returns:
        np.ndarray: Complex coefficients [a_0, a_1, ..., a_d].
    """
    # Sample real and imaginary parts
    real = rng.uniform(-1.0, 1.0, size=degree + 1)
    imag = rng.uniform(-1.0, 1.0, size=degree + 1)
    coeffs = real + 1j * imag
    
    # Ensure leading coefficient is non-zero
    while np.abs(coeffs[degree]) < 0.1:
        r = rng.uniform(-1.0, 1.0)
        i = rng.uniform(-1.0, 1.0)
        coeffs[degree] = r + 1j * i
        
    return coeffs

def evaluate_polynomial(coeffs: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Evaluate the polynomial with given coefficients on a grid of complex values.

    p(z) = a_0 + a_1*z + a_2*z^2 + ... + a_d*z^d

    Args:
        coeffs: 1D array of complex coefficients of shape (d+1,).
        z: NumPy array of complex numbers to evaluate.

    Returns:
        np.ndarray: The evaluated polynomial values, same shape as z.
    """
    # We use Horner's method for efficient evaluation
    res = np.zeros_like(z, dtype=complex)
    for c in reversed(coeffs):
        res = res * z + c
    return res

def apply_label_noise(
    true_degree: int, 
    max_degree: int, 
    label_noise: float, 
    rng: np.random.Generator
) -> int:
    """
    Apply label noise to the polynomial degree.
    
    With probability label_noise, returns a random degree in [1, max_degree]
    that is different from the true degree. Otherwise, returns the true degree.

    Args:
        true_degree: The true degree of the polynomial.
        max_degree: The maximum possible degree.
        label_noise: Probability of replacing the true label.
        rng: A NumPy random Generator.

    Returns:
        int: The (possibly noisy) degree.
    """
    if max_degree <= 1 or label_noise <= 0.0:
        return true_degree
        
    if rng.uniform(0.0, 1.0) < label_noise:
        # Choose a different degree
        possible_labels = [d for d in range(1, max_degree + 1) if d != true_degree]
        if possible_labels:
            return int(rng.choice(possible_labels))
            
    return true_degree
