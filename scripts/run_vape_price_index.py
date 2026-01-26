# -*- coding: utf-8 -*-
"""
Runs the vape_price_index.py library
"""

import argparse
from pathlib import Path

from src.vape_price_index import (
    process_all_stores, 
    build_panel_index,
    )


def parse_args() -> argparse.Namespace:

    # create a parser object (like a spec sheet for my program) defining what arguments my program can take
    parser = argparse.ArgumentParser(
        description="Run vape price index pipeline."
    )

    # each call to `parser.add_argument(...)` adds one allowable parameter to the spec.
    parser.add_argument(
        "--store-path",
        type=str,
        required=True,
        help="Directory containing raw per-store feather files."
    )

    parser.add_argument(
        "--outpath",
        type=str,
        required=True,
        help="Directory where per-store vape index feather files will be written."
    )

    parser.add_argument(
        "--panel-output-path",
        type=str,
        default=None,
        help="Optional output feather for the panel index file."
    )

    parser.add_argument(
        "--weight-basis",
        choices=["fiscal", "calendar"],
        default="fiscal",
        help=(
            "Revenue-weighting basis for annual weights. "
            "'fiscal' uses fiscal_year; 'calendar' uses calendar_year."
            ),
    )

    parser.add_argument(
        "--index-kind",
        choices=["price", "qty"],
        default="price",
        help="Which index to construct: 'price' uses unit_value_q; 'qty' uses quantity.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N stores (dry run)."
    )

    # parse the command-line arguments according to the spec sheet (i.e. "turn the text into variables")
    return parser.parse_args()


def main() -> None:
    # parse command-line arguments (strings to variables)
    args = parse_args()

    # path normalization (convert str to pathlib.path), required for mkdir below
    store_path = Path(args.store_path)
    outpath = Path(args.outpath)

    outpath.mkdir(parents=True, exist_ok=True)

    """
    Process (possibly limited) number of stores
    Note: This is where the CLI's first-class parameters become function arguments 
    that control computation.
    """
    process_all_stores(
        store_path=str(store_path), # convert Path back to str
        outpath=str(outpath),
        weight_basis=args.weight_basis,
        index_kind=args.index_kind,
        limit=args.limit
    )

    # Optional: build the panel
    if args.panel_output_path is not None:
        panel_output_path = Path(args.panel_output_path)
        panel_output_path.parent.mkdir(parents=True, exist_ok=True)

        build_panel_index(
            source_dir=str(outpath),
            output_path=str(panel_output_path)
        )


if __name__ == "__main__":
    main()



# rootdir = r"C:/Users/cahase/Documents/e_cigs_best_practices/"

# def main() -> None:
#     # Adjust these as needed
#     store_path = rootdir + "data/interim/da_store_id_monthly_ag/"
#     outpath = rootdir + "data/processed/store_vape_price_indexes/"
#     panel_output_path = rootdir + "data/processed/index_panels/vape_price_indexes.feather"

#     process_all_stores(store_path=str(store_path), outpath=str(outpath))
#     build_panel_index(source_dir=str(outpath), output_path=str(panel_output_path))


# if __name__ == "__main__":
#     main()
    
    
