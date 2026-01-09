"""Image retrieval system using CLIP embeddings and FAISS."""

import argparse
import logging
import sys
from pathlib import Path

from .config import default_config
from .indexer import ImageIndexer
from .search import SearchEngine
from .server import run_server
from .split_image import split_image
from .utils import get_logger, print_header, setup_logging

__version__ = "0.1.0"


def cmd_split(args: argparse.Namespace) -> int:
    """Execute the split command."""
    logger = get_logger()

    try:
        print_header("DATASET SPLITTER")

        split_image(
            image_folder=args.input,
            train_folder=args.train,
            test_folder=args.test,
            ratio=args.ratio,
            state=args.seed,
            copy=args.copy,
        )

        logger.info("Dataset split completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Failed to split dataset: {e}", exc_info=args.verbose)
        return 1


def cmd_index(args: argparse.Namespace) -> int:
    """Execute the index command."""
    logger = get_logger()

    try:
        print_header("IMAGE INDEXER")

        config = default_config
        if args.batch_size:
            config.processing.batch_size = args.batch_size

        indexer = ImageIndexer(config)
        indexer.create_index(
            image_folder=Path(args.input),
            index_folder=Path(args.output),
            batch_size=args.batch_size,
        )

        logger.info("Indexing completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Failed to create index: {e}", exc_info=args.verbose)
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Execute the search command."""
    logger = get_logger()

    try:
        config = default_config
        search_engine = SearchEngine(args.index, config)

        if args.text:
            # Text search
            print(f"\nSearching for: '{args.text}'")
            results = search_engine.search_by_text(args.text, k=args.k)

        elif args.image:
            # Image search
            print(f"\nSearching for similar images to: {args.image}")
            results = search_engine.search_by_image(args.image, k=args.k)

        else:
            logger.error("Must specify either --text or --image")
            return 1

        # Display results
        print(f"\nFound {len(results)} results:\n")
        for result in results:
            print(f"  [{result['rank']}] {result['file']}")
            print(f"      Score: {result['score']:.4f}")

        # Save results if requested
        if args.output:
            import json

            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=args.verbose)
        return 1


def cmd_server(args: argparse.Namespace) -> int:
    """Execute the server command."""
    logger = get_logger()

    try:
        run_server(host=args.host, port=args.port)
        return 0

    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=args.verbose)
        return 1


def cmd_stats(args: argparse.Namespace) -> int:
    """Execute the stats command."""
    logger = get_logger()

    try:
        config = default_config
        search_engine = SearchEngine(args.index, config)
        stats = search_engine.get_stats()

        print_header("INDEX STATISTICS")
        print()
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print()

        return 0

    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=args.verbose)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="image-rel",
        description="Image retrieval system using CLIP and FAISS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        required=True,
    )
    server_parser = subparsers.add_parser(
        "server",
        help="Start the server",
    )
    server_parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    server_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to listen on (default: 0.0.0.0)",
    )
    server_parser.set_defaults(func=cmd_server)

    # Split command
    split_parser = subparsers.add_parser(
        "split",
        help="Split images into train and test sets",
    )
    split_parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/archive",
        help="Input folder containing images (default: data/archive)",
    )
    split_parser.add_argument(
        "--train",
        type=str,
        default="data/train",
        help="Output folder for training images (default: data/train)",
    )
    split_parser.add_argument(
        "--test",
        type=str,
        default="data/test",
        help="Output folder for test images (default: data/test)",
    )
    split_parser.add_argument(
        "-r",
        "--ratio",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    split_parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    split_parser.add_argument(
        "-c",
        "--copy",
        action="store_true",
        help="Copy files instead of moving them",
    )
    split_parser.set_defaults(func=cmd_split)

    # Index command
    index_parser = subparsers.add_parser(
        "index",
        help="Create FAISS index from images",
    )
    index_parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/train",
        help="Input folder containing images (default: data/train)",
    )
    index_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/index",
        help="Output folder for index (default: data/index)",
    )
    index_parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for processing (default: 32)",
    )
    index_parser.set_defaults(func=cmd_index)

    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Search for similar images",
    )
    search_parser.add_argument(
        "--index",
        type=str,
        default="data/index",
        help="Index folder (default: data/index)",
    )
    search_group = search_parser.add_mutually_exclusive_group(required=True)
    search_group.add_argument(
        "-t",
        "--text",
        type=str,
        help="Text query for searching images",
    )
    search_group.add_argument(
        "-i",
        "--image",
        type=str,
        help="Image path for similarity search",
    )
    search_parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    search_parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file to save results (JSON format)",
    )
    search_parser.set_defaults(func=cmd_search)

    # Stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show index statistics",
    )
    stats_parser.add_argument(
        "--index",
        type=str,
        default="data/index",
        help="Index folder (default: data/index)",
    )
    stats_parser.set_defaults(func=cmd_stats)

    return parser


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # Execute command
    try:
        exit_code = args.func(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(130)
    except Exception as e:
        logger = get_logger()
        logger.error(f"Unexpected error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
