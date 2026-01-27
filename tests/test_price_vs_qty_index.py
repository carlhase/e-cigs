import numpy as np
import pandas as pd

from src.vape_price_index import compute_vape_price_index_for_store


def test_price_and_quantity_indexes_differ():
    """
    Sanity check: price and quantity indexes should not be identical
    when unit_value_q and quantity evolve differently.

    This test guards against accidentally reusing price logic
    when computing quantity indexes.
    """

    # Minimal synthetic store–product–month panel
    df = pd.DataFrame(
        {
            "store_id": ["1", "1", "1", "1"],
            "gtin": ["A", "A", "A", "A"],               # gtin is str type
            "subcategory": ["Vaping Products"] * 4,
            "product_type": ["Pods"] * 4,
            "calendar_year": [2023, 2023, 2023, 2023],
            "calendar_month": [1, 2, 3, 4],
            "date": pd.period_range("2023-01", periods=4, freq="M"),
            "fiscal_year": [2023, 2023, 2023, 2023],
            "unit_value_q": [10.0, 12.0, 12.0, 15.0],   # price path
            "quantity": [100, 120, 90, 110],            # quantity path (different)
            "total_revenue_amount": [1000, 1440, 1080, 1650],
            "scan_type": ["GTIN"] * 4,
        }
    )

    price_index = compute_vape_price_index_for_store(
        df,
        weight_basis="calendar",
        index_kind="price",
    )

    qty_index = compute_vape_price_index_for_store(
        df,
        weight_basis="calendar",
        index_kind="qty",
    )

    # Extract comparable series
    p = price_index["vape_price_index"]
    q = qty_index["vape_qty_index"]

    # Core invariant: price and quantity indexes must not coincide
    assert not np.allclose(
        p.values,
        q.values,
        equal_nan=True,
    ), "Price and quantity indexes should differ"
