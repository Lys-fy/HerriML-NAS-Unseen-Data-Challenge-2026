import numpy as np

class Domino:
    def __init__(
        self,
        row: int,
        col: int,
        val1: int,
        val2: int,
        orientation: str
    ):
        """
        Represents a domino piece.

        Args:
            row: Grid row index (used internally by path generator).
            col: Grid column index (used internally by path generator).
            val1: Value of the left/top half (input side, connects to predecessor).
            val2: Value of the right/bottom half (output side, connects to successor).
            orientation: 'H' for horizontal, 'V' for vertical (unused in strip render).
        """
        self.row = row
        self.col = col
        self.val1 = val1
        self.val2 = val2
        self.orientation = orientation


def find_self_avoiding_walk(
    n: int,
    grid_rows: int,
    grid_cols: int,
    rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    """
    Find a self-avoiding walk of length n on a grid.

    Args:
        n: Number of steps (dominoes).
        grid_rows: Grid height.
        grid_cols: Grid width.
        rng: NumPy random Generator.

    Returns:
        list of (row, col) positions of length n, or None on failure.
    """
    def get_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_rows and 0 <= nc < grid_cols:
                result.append((nr, nc))
        return result

    def dfs(path: list[tuple[int, int]], visited: set) -> list[tuple[int, int]] | None:
        if len(path) == n:
            return path
        r, c = path[-1]
        neighbors = get_neighbors(r, c)
        rng.shuffle(neighbors)
        for nr, nc in neighbors:
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                path.append((nr, nc))
                result = dfs(path, visited)
                if result is not None:
                    return result
                path.pop()
                visited.remove((nr, nc))
        return None

    starts = [(r, c) for r in range(grid_rows) for c in range(grid_cols)]
    rng.shuffle(starts)
    for sr, sc in starts:
        path = dfs([(sr, sc)], {(sr, sc)})
        if path is not None:
            return path
    return None


def generate_domino_chain(
    n: int,
    rng: np.random.Generator,
    inject_error: bool = False,
    error_boundary: int | None = None
) -> tuple[list[Domino], int]:
    """
    Generate n dominoes forming a chain, returned in chain order.

    The chain rule: domino[i].val2 == domino[i+1].val1 for all i in [0, n-2].
    If inject_error is True, exactly one adjacent pair will violate this rule.

    Args:
        n: Number of dominoes.
        rng: NumPy random Generator.
        inject_error: If True, inject exactly one mismatch.
        error_boundary: If given (and inject_error), inject the mismatch at this
            specific junction index 0..n-2 (used to balance the localisation
            task). If None, a random junction is chosen.

    Returns:
        (list[Domino] in chain order, boundary): boundary is the junction index
        of the injected mismatch, or -1 for a valid chain.
    """
    grid_dim = int(np.ceil(np.sqrt(n))) + 1
    path = find_self_avoiding_walk(n, grid_dim, grid_dim, rng)
    if path is None:
        path = find_self_avoiding_walk(n, grid_dim + 1, grid_dim + 1, rng)
        if path is None:
            raise ValueError(f"Could not generate self-avoiding walk of length {n}.")

    # Determine orientation from step direction
    orientations = []
    for i in range(n - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        orientations.append('H' if r1 == r2 else 'V')
    orientations.append(orientations[-1] if orientations else 'H')

    # Build a flat value array: [val1_0, val2_0, val1_1, val2_1, ...]
    # Chain rule: vals[2*i + 1] == vals[2*(i+1)] for a valid chain
    vals: list[int] = []
    vals.append(int(rng.integers(0, 7)))   # val1 of domino 0
    vals.append(int(rng.integers(0, 7)))   # val2 of domino 0
    for i in range(1, n):
        # val1 of domino i must equal val2 of domino i-1 (for valid chain)
        vals.append(vals[-1])              # val1 = previous val2
        vals.append(int(rng.integers(0, 7)))  # val2 is free

    # Inject exactly one mismatch at the chosen (or a random) boundary
    used_boundary = -1
    if inject_error and n > 1:
        if error_boundary is None:
            boundary = int(rng.integers(0, n - 1))
        else:
            boundary = int(error_boundary)
        correct_val1 = vals[2 * (boundary + 1)]
        wrong_val1 = int(rng.integers(0, 7))
        while wrong_val1 == correct_val1:
            wrong_val1 = int(rng.integers(0, 7))
        vals[2 * (boundary + 1)] = wrong_val1
        used_boundary = boundary

    # Construct Domino objects in chain order
    dominoes = []
    for i in range(n):
        r, c = path[i]
        dominoes.append(Domino(r, c, vals[2 * i], vals[2 * i + 1], orientations[i]))

    return dominoes, used_boundary


def apply_label_noise(
    true_label: int,
    label_noise: float,
    rng: np.random.Generator
) -> int:
    """
    Flip the binary validity label with probability label_noise.

    Args:
        true_label: 0 (valid) or 1 (invalid).
        label_noise: Probability of flipping.
        rng: NumPy random Generator.

    Returns:
        int: Possibly flipped label.
    """
    if label_noise <= 0.0:
        return true_label
    if rng.uniform(0.0, 1.0) < label_noise:
        return 1 - true_label
    return true_label


def apply_label_noise_multiclass(
    true_label: int,
    n_classes: int,
    label_noise: float,
    rng: np.random.Generator
) -> int:
    """
    With probability label_noise, replace the label with a different random
    class in 0..n_classes-1 (used by the localisation task).
    """
    if label_noise <= 0.0 or n_classes < 2:
        return true_label
    if rng.uniform(0.0, 1.0) < label_noise:
        other = int(rng.integers(0, n_classes - 1))
        if other >= true_label:
            other += 1
        return other
    return true_label
