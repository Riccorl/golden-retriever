import argparse
from goldenretriever.common.data_utils import preprocess_to_mds


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input", type=str, help="Path to the input file.")
    arg_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Path to the cache directory.",
    )
    arg_parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Whether to use cache.",
    )
    arg_parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers to use.",
    )
    args = arg_parser.parse_args()

    preprocess_to_mds(
        source=args.input,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        num_workers=args.num_workers,
    )
