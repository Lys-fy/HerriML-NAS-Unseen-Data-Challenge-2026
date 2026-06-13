import numpy as np
from shared.random import get_rng
from shared.noise import apply_image_noise
from shared.io import save_dataset
from connect4.utils import (
    init_board,
    get_lowest_empty_row,
    make_move,
    check_win,
    get_winning_columns,
    apply_label_noise
)
from connect4.render import render_board

def generate_dataset(
    num_samples: int,
    image_size: int = 64,
    noise_level: float = 0.0,
    label_noise: float = 0.0,
    color_mode: str = "fixed",
    seed: int | None = None,
    output_dir: str = "./dataset",
    export_hdf5: bool = False,
    board_width: int = 10,
    board_height: int = 10,
    connect_k: int = 4
) -> None:
    """
    Generate the Connect Four Win Exists dataset.

    Simulates legal games of Connect Four, extracts board states, checks for 
    immediate winning columns, renders board states as RGB images, applies 
    noise, and exports the data.

    Args:
        num_samples: Number of samples to generate.
        image_size: Size of the square RGB images.
        noise_level: Standard deviation factor for image noise.
        label_noise: Probability of replacing the true label with a noisy label.
        color_mode: Color mode ('fixed', 'per_sample', 'random_all').
        seed: Random seed for reproducibility.
        output_dir: Target output directory.
        export_hdf5: Whether to export datasets in HDF5 format.
        board_width: Number of columns on the board.
        board_height: Number of rows on the board.
        connect_k: Number of consecutive pieces required to win.
    """
    rng = get_rng(seed)
    
    images = []
    labels = []
    
    win_pool = []
    no_win_pool = []
    
    attempts = 0
    max_attempts = num_samples * 20
    
    # Determine the number of samples we want for each class to balance the dataset
    half_needed = num_samples // 2
    
    while (len(win_pool) < half_needed or len(no_win_pool) < half_needed) and attempts < max_attempts:
        attempts += 1
        
        # Start a new game
        board = init_board(board_height, board_width)
        
        max_possible_moves = board_height * board_width
        min_moves = connect_k * 2 - 2
        if min_moves >= max_possible_moves:
            min_moves = 0
            
        # Decide a random number of moves to play
        num_moves = rng.integers(min_moves, max_possible_moves - 1) if min_moves < max_possible_moves - 1 else min_moves
        
        game_over = False
        valid_history = []
        
        for move_idx in range(num_moves):
            legal_cols = [c for c in range(board_width) if get_lowest_empty_row(board, c) is not None]
            
            if not legal_cols:
                break
                
            col = rng.choice(legal_cols)
            board, _ = make_move(board, col)
            
            # Since make_move swaps channels, the player who just moved is now Channel 0
            if check_win(board, 0, connect_k):
                game_over = True
                break
                
            if move_idx >= min_moves:
                valid_history.append(board.copy())
                
        if game_over or not valid_history:
            continue
            
        # Pick a random state from history
        sample_board = valid_history[rng.integers(0, len(valid_history))]
        
        # Check winning moves for the active player (Channel 1)
        winning_cols = get_winning_columns(sample_board, connect_k)
        if winning_cols:
            true_label = min(winning_cols)
            if len(win_pool) < num_samples:
                win_pool.append((sample_board, true_label))
        else:
            true_label = board_width
            if len(no_win_pool) < num_samples:
                no_win_pool.append((sample_board, true_label))
                
    # Combine pools to select final set of samples
    selected_samples = []
    
    if len(win_pool) >= half_needed and len(no_win_pool) >= half_needed:
        rng.shuffle(win_pool)
        rng.shuffle(no_win_pool)
        selected_samples.extend(win_pool[:half_needed])
        selected_samples.extend(no_win_pool[:(num_samples - half_needed)])
    else:
        combined = win_pool + no_win_pool
        rng.shuffle(combined)
        selected_samples = combined[:num_samples]
        
    rng.shuffle(selected_samples)
    
    # Process and save selected samples
    for board_state, true_label in selected_samples:
        img = render_board(board_state, image_size, color_mode, rng)
        img_noisy = apply_image_noise(img, noise_level, rng)
        
        final_label = apply_label_noise(true_label, board_width, label_noise, rng)
        
        images.append(img_noisy)
        labels.append({
            "winning_column": final_label,
            "true_winning_column": true_label
        })
        
    save_dataset(
        images=images,
        labels=labels,
        output_dir=output_dir,
        export_hdf5=export_hdf5,
        rng=rng
    )
