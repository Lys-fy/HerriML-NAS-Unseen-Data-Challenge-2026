import numpy as np

def init_board(board_height: int, board_width: int) -> np.ndarray:
    """
    Initialize an empty Connect Four board.
    
    The board is represented as a board_height x board_width x 2 array:
    - Axis 2, Index 0 (Channel 0): Inactive player's pieces (color 1).
    - Axis 2, Index 1 (Channel 1): Active player's pieces (color 2).
    - It is always the active player's turn to play.

    Args:
        board_height: Number of rows.
        board_width: Number of columns.

    Returns:
        np.ndarray: A zero-initialized array of shape (board_height, board_width, 2).
    """
    return np.zeros((board_height, board_width, 2), dtype=np.uint8)

def get_lowest_empty_row(board: np.ndarray, col: int) -> int | None:
    """
    Find the lowest empty row in a given column.
    
    Row 0 is the bottom row, and H-1 is the top row.

    Args:
        board: Board state of shape (H, W, 2).
        col: The column index.

    Returns:
        int: The row index, or None if the column is full.
    """
    h = board.shape[0]
    occupied = board[:, col, 0] | board[:, col, 1]
    for r in range(h):
        if not occupied[r]:
            return r
    return None

def make_move(board: np.ndarray, col: int) -> tuple[np.ndarray, int]:
    """
    Place a piece for the active player (Channel 1) in the given column.
    
    Since the turn alternates, the active player becomes inactive (Channel 0),
    and the inactive player becomes the active player (Channel 1).

    Args:
        board: Board state of shape (H, W, 2).
        col: Column to drop the piece in.

    Returns:
        tuple[np.ndarray, int]: New board state and the row index where the piece fell.
    """
    row = get_lowest_empty_row(board, col)
    if row is None:
        raise ValueError(f"Column {col} is full!")
    
    new_board = np.zeros_like(board)
    # Channel Swap: Old inactive (0) becomes new active (1)
    new_board[:, :, 1] = board[:, :, 0]
    # Old active (1) becomes new inactive (0)
    new_board[:, :, 0] = board[:, :, 1]
    
    # Place the new piece for the player who just moved (now inactive at index 0)
    new_board[row, col, 0] = 1
    return new_board, row

def check_win(board: np.ndarray, player: int, connect_k: int) -> bool:
    """
    Check if the given player has won the game (connect_k in a row).

    Args:
        board: Board state of shape (H, W, 2).
        player: Player channel index (0 or 1).
        connect_k: Number of consecutive pieces required to win.

    Returns:
        bool: True if the player has won, False otherwise.
    """
    ch = board[:, :, player]
    h, w = ch.shape
    
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for r in range(h):
        for c in range(w):
            if ch[r, c] == 0:
                continue
                
            for dr, dc in directions:
                if 0 <= r + (connect_k - 1) * dr < h and 0 <= c + (connect_k - 1) * dc < w:
                    match = True
                    for step in range(1, connect_k):
                        if ch[r + step * dr, c + step * dc] == 0:
                            match = False
                            break
                    if match:
                        return True
    return False

def get_winning_columns(board: np.ndarray, connect_k: int) -> list[int]:
    """
    Check all legal moves for the active player (Channel 1) and return columns
    that result in an immediate win.

    Args:
        board: Board state of shape (H, W, 2).
        connect_k: Number of consecutive pieces required to win.

    Returns:
        list[int]: Column indices that lead to an immediate win for Channel 1.
    """
    winning_cols = []
    w = board.shape[1]
    
    for col in range(w):
        row = get_lowest_empty_row(board, col)
        if row is not None:
            # Temporarily place stone in Channel 1 (active player)
            board[row, col, 1] = 1
            if check_win(board, 1, connect_k):
                winning_cols.append(col)
            # Revert
            board[row, col, 1] = 0
                
    return winning_cols

def apply_label_noise(
    true_label: int, 
    board_width: int, 
    label_noise: float, 
    rng: np.random.Generator
) -> int:
    """
    Apply label noise to the winning column label.
    """
    if label_noise <= 0.0:
        return true_label
        
    if rng.uniform(0.0, 1.0) < label_noise:
        choices = [c for c in range(board_width + 1) if c != true_label]
        return int(rng.choice(choices))
        
    return true_label
