import numpy as np
import pandas as pd

from src.vape_price_index import compute_vape_price_index_for_store

def test_fiscal_and_calendar_indexes_differ():
    """
    Sanity check: Fiscal- and calendar-weighted indexes should differ when:
      - fiscal year spans two calendar years, and
      - the product revenue mix changes across calendar years.

      This guards against ignoring weight_basis.
    """

    # Minimal synthetic store–product–month panel
    months = pd.period_range("2022-10", periods=6, freq="M")

    base = pd.DataFrame(
        {
            "store_id": ["1"] * 6,
            "subcategory": ["Vaping Products"] * 6,
            "product_type": ["Pods"] * 6,
            "calendar_year": [2022, 2022, 2022, 2023, 2023, 2023],
            "calendar_month": [10, 11, 12, 1, 2, 3],
            "date": months,           # length 6
            "fiscal_year": [2023] * 6,
            "scan_type": ["GTIN"] * 6,
        }
    )

    # Two GTINs in one product_type, with a revenue-mix shift across calendar years:
    # - In 2022 (Oct-Dec), GTIN A dominates revenue
    # - In 2023 (Jan-Mar), GTIN B dominates revenue#
    # Prices evolve differently for A vs B, so changing weights should change the index.
    
    # GTIN A: rising prices; revenue high in 2022, low in 2023
    df_a = base.copy()
    df_a["gtin"] = "A"
    df_a["unit_value_q"] = [10, 11, 12, 13, 14, 15]
    df_a["quantity"] = [100] * 6
    df_a["total_revenue_amount"] = [1000, 1000, 1000, 100, 100, 100]

    # GTIN B: falling prices; revenue low in 2022, high in 2023
    df_b = base.copy()
    df_b["gtin"] = "B"
    df_b["unit_value_q"] = [20, 20, 20, 19, 18, 17]
    df_b["quantity"] = [100] * 6
    df_b["total_revenue_amount"] = [100, 100, 100, 1000, 1000, 1000]

    df = pd.concat([df_a, df_b], ignore_index=True)

    fiscal_index = compute_vape_price_index_for_store(
        df,
        weight_basis="fiscal",
        index_kind="price",
    )

    calendar_index = compute_vape_price_index_for_store(
        df,
        weight_basis="calendar",
        index_kind="price",
    )

    # Extract comparable series
    f = fiscal_index["vape_price_index"].to_numpy()
    c = calendar_index["vape_price_index"].to_numpy()

    # Core invariant: fiscal and calendar indexes must not coincide
    assert not np.allclose(f, c, equal_nan=True), (
        "Fiscal and calendar indexes should differ when the revenue mix shifts "
        "across calendar years inside one fiscal year."
    )