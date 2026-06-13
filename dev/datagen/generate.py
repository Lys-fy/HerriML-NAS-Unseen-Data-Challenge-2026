import argparse
import sys

def main() -> None:
    """
    Command Line Interface entrypoint for generating the synthetic datasets.
    """
    parser = argparse.ArgumentParser(
        description="Python Synthetic Dataset Generator Project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Shared arguments
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["mathvision", "connect4", "domino", "polygon_collision"], 
        help="Which synthetic dataset to generate"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        required=True, 
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=64, 
        help="Size of the output square RGB images"
    )
    parser.add_argument(
        "--noise_level", 
        type=float, 
        default=0.0, 
        help="Standard deviation factor for image noise [0.0 - 1.0]"
    )
    parser.add_argument(
        "--label_noise", 
        type=float, 
        default=0.0, 
        help="Probability of intentionally corrupting labels [0.0 - 1.0]"
    )
    parser.add_argument(
        "--color_mode", 
        type=str, 
        default="fixed", 
        choices=["fixed", "per_sample", "random_all"], 
        help="Aesthetic color mode for generators"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Deterministic random seed"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./dataset", 
        help="Output target directory"
    )
    parser.add_argument(
        "--export_hdf5", 
        action="store_true", 
        help="Whether to export the datasets in HDF5 format"
    )
    
    # Dataset 1: MathVision arguments
    parser.add_argument(
        "--max_degree", 
        type=int, 
        default=5, 
        help="[MathVision] Maximum polynomial degree"
    )
    
    # Dataset 2: Connect Four arguments
    parser.add_argument(
        "--board_width", 
        type=int, 
        default=10, 
        help="[Connect4] Width of the board"
    )
    parser.add_argument(
        "--board_height", 
        type=int, 
        default=10, 
        help="[Connect4] Height of the board"
    )
    parser.add_argument(
        "--connect_k", 
        type=int, 
        default=4, 
        help="[Connect4] Win condition length"
    )
    
    # Dataset 3: Domino arguments
    parser.add_argument(
        "--min_dominoes",
        type=int,
        default=4,
        help="[Domino] Minimum number of domino tiles per chain"
    )
    parser.add_argument(
        "--max_dominoes",
        type=int,
        default=9,
        help="[Domino] Maximum number of domino tiles per chain (inclusive)"
    )
    parser.add_argument(
        "--domino_task",
        type=str,
        default="localisation",
        choices=["localisation", "validity"],
        help="[Domino] 'localisation' = predict the mismatch junction; "
             "'validity' = binary valid/invalid"
    )
    parser.add_argument(
        "--domino_count",
        type=int,
        default=5,
        help="[Domino] Fixed chain length for the localisation task"
    )
    parser.add_argument(
        "--domino_tile",
        type=int,
        default=34,
        help="[Domino] Half-cell size in px (pip readability)"
    )

    # Dataset 4: Polygon Collision arguments
    parser.add_argument(
        "--num_sides", 
        type=int, 
        default=5, 
        help="[Polygon Collision] Number of sides of the polygon"
    )
    parser.add_argument(
        "--num_timesteps", 
        type=int, 
        default=4, 
        help="[Polygon Collision] Number of rendered timesteps"
    )
    parser.add_argument(
        "--regular", 
        action="store_true", 
        help="[Polygon Collision] Create a regular polygon (otherwise irregular)"
    )
    parser.add_argument(
        "--edges_different_color", 
        action="store_true", 
        help="[Polygon Collision] Color polygon edges differently"
    )
    parser.add_argument(
        "--poly_color_random", 
        action="store_true", 
        help="[Polygon Collision] Randomize polygon colors across samples"
    )
    parser.add_argument(
        "--obj_color_random", 
        action="store_true", 
        help="[Polygon Collision] Randomize object colors across samples"
    )
    parser.add_argument(
        "--obj_color_random_timesteps", 
        action="store_true", 
        help="[Polygon Collision] Randomize object colors within timesteps"
    )
    
    args = parser.parse_args()
    
    # Invoke corresponding generator
    if args.dataset == "mathvision":
        from mathvision.generator import generate_dataset as math_gen
        math_gen(
            num_samples=args.num_samples,
            image_size=args.image_size,
            noise_level=args.noise_level,
            label_noise=args.label_noise,
            color_mode=args.color_mode,
            seed=args.seed,
            output_dir=args.output_dir,
            export_hdf5=args.export_hdf5,
            max_degree=args.max_degree
        )
    elif args.dataset == "connect4":
        from connect4.generator import generate_dataset as c4_gen
        c4_gen(
            num_samples=args.num_samples,
            image_size=args.image_size,
            noise_level=args.noise_level,
            label_noise=args.label_noise,
            color_mode=args.color_mode,
            seed=args.seed,
            output_dir=args.output_dir,
            export_hdf5=args.export_hdf5,
            board_width=args.board_width,
            board_height=args.board_height,
            connect_k=args.connect_k
        )
    elif args.dataset == "domino":
        from domino.generator import generate_dataset as domino_gen
        domino_gen(
            num_samples=args.num_samples,
            image_size=args.image_size,
            noise_level=args.noise_level,
            label_noise=args.label_noise,
            color_mode=args.color_mode,
            seed=args.seed,
            output_dir=args.output_dir,
            export_hdf5=args.export_hdf5,
            min_dominoes=args.min_dominoes,
            max_dominoes=args.max_dominoes,
            task=args.domino_task,
            domino_count=args.domino_count,
            tile=args.domino_tile,
        )
    elif args.dataset == "polygon_collision":
        from polygon_collision.generator import generate_dataset as poly_gen
        poly_gen(
            num_samples=args.num_samples,
            image_size=args.image_size,
            noise_level=args.noise_level,
            label_noise=args.label_noise,
            color_mode=args.color_mode,
            seed=args.seed,
            output_dir=args.output_dir,
            export_hdf5=args.export_hdf5,
            num_sides=args.num_sides,
            num_timesteps=args.num_timesteps,
            regular=args.regular,
            edges_same_color=not args.edges_different_color,
            poly_color_fixed=not args.poly_color_random,
            obj_color_fixed=not args.obj_color_random,
            obj_color_fixed_within_timesteps=not args.obj_color_random_timesteps
        )
        
    print(f"Successfully generated {args.num_samples} samples for dataset '{args.dataset}' in '{args.output_dir}'.")

if __name__ == "__main__":
    main()
