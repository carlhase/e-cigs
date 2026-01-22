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
    parser = argparse.ArgumentParser(
        description="Run vape price index pipeline."
    )

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
        "--limit",
        type=int,
        default=None,
        help="Process only the first N stores (dry run)."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    store_path = Path(args.store_path)
    outpath = Path(args.outpath)

    outpath.mkdir(parents=True, exist_ok=True)

    # Process (possibly limited) number of stores
    process_all_stores(
        store_path=str(store_path),
        outpath=str(outpath),
        weight_basis=args.weight_basis,
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
    
    
