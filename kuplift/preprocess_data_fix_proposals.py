"""
These are fix proposals for the `preprocess_data` function found in *helperfunctions.py*.
(in case the probability computation is really erroneous)

There are two versions in this file:
- `preprocess_data_fast`
- `preprocess_data_faster`.
Choose one if it fits, rename it to `preprocess_data` and move the code to *helperfunctions.py*.
These versions improve a lot on robustness, so the code in greatly modified. If this is not
wanted, just fix the two occurrences of the computation directly in the original function.
"""


import pandas as pd
import numpy as np


def preprocess_data_fast(data, treatment_col="segment", y_col="visit"):
    """
    Vectorized and backward-compatible version of preprocess_data.

    Preserved compatibility:
    - same logic for selecting numerical columns (>= 1000 unique values)
    - other columns treated as categorical
    - categorical encoding ordered by increasing uplift
    - categorical NaNs -> "NAN_VAL" then encoded as -1
    - final cast of treatment_col to str

    Included fix:
    - P(Y=1|T=0) computed as (t0y1 / (t0y0 + t0y1))
      (and not t0y1 / (t0y1 + t0y1))
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame")
    if treatment_col not in data.columns:
        raise ValueError(f"`treatment_col`='{treatment_col}' not found in dataframe")
    if y_col not in data.columns:
        raise ValueError(f"`y_col`='{y_col}' not found in dataframe")

    df = data.copy()

    # 1) Numerical columns (as in the legacy code)
    cols = df.columns.tolist()
    num_cols = df._get_numeric_data().columns.tolist()

    if treatment_col in num_cols:
        num_cols.remove(treatment_col)
    if y_col in num_cols:
        num_cols.remove(y_col)

    # Keep original behavior: threshold of 1000 unique values
    kept_num_cols = []
    for col in num_cols:
        if df[col].nunique(dropna=True) >= 1000:
            col_min = df[col].min(skipna=True)
            fill_val = -1 if pd.isna(col_min) else (col_min - 1)
            df[col] = df[col].fillna(fill_val)
            kept_num_cols.append(col)

    # 2) Categorical columns = everything else except treatment/outcome
    categorical_cols = list(set(cols) - set(kept_num_cols))
    if treatment_col in categorical_cols:
        categorical_cols.remove(treatment_col)
    if y_col in categorical_cols:
        categorical_cols.remove(y_col)

     # 3) Vectorized uplift encoding
    for cat_col in categorical_cols:
        # Same convention as the legacy code
        s = df[cat_col].fillna("NAN_VAL")

        # Compact table
        tmp = pd.DataFrame({
            cat_col: s,
            treatment_col: df[treatment_col],
            y_col: df[y_col]
        })

        # Ignore NAN_VAL to reproduce original logic
        tmp_non_missing = tmp[tmp[cat_col] != "NAN_VAL"]

        if tmp_non_missing.empty:
            # All values missing: everything is set to -1
            df[cat_col] = -1
            continue

        # Vectorized counts per category:
        # t0y0, t0y1, t1y0, t1y1
        grp = tmp_non_missing.groupby(cat_col, dropna=False)

        t0y0 = grp.apply(lambda g: ((g[treatment_col] == 0) & (g[y_col] == 0)).sum())
        t0y1 = grp.apply(lambda g: ((g[treatment_col] == 0) & (g[y_col] == 1)).sum())
        t1y0 = grp.apply(lambda g: ((g[treatment_col] == 1) & (g[y_col] == 0)).sum())
        t1y1 = grp.apply(lambda g: ((g[treatment_col] == 1) & (g[y_col] == 1)).sum())

        counts = pd.DataFrame({
            "t0y0": t0y0,
            "t0y1": t0y1,
            "t1y0": t1y0,
            "t1y1": t1y1
        })

        den_t = counts["t1y1"] + counts["t1y0"]
        den_c = counts["t0y1"] + counts["t0y0"]  # <-- fix

        # Robust probability computation
        p_t = counts["t1y1"] / den_t.replace(0, np.nan)
        p_c = counts["t0y1"] / den_c.replace(0, np.nan)

        # Fallbacks compatible with previous behavior
        uplift = p_t - p_c
        uplift = uplift.where(~p_t.isna(), -1.0)  # no treatment
        uplift = uplift.where(~p_c.isna(), 0.0)   # no control

        # Ascending sort (as before)
        # Keep a deterministic tie-break using stringified index
        order_df = pd.DataFrame({"uplift": uplift})
        order_df["_k"] = order_df.index.astype(str)
        order_df = order_df.sort_values(["uplift", "_k"], ascending=[True, True])

        value_to_code = {val: i for i, val in enumerate(order_df.index.tolist())}

        encoded = s.map(value_to_code).fillna(-1).astype(int)
        df[cat_col] = encoded

    # 4) Strict compatibility with the legacy code
    df[treatment_col] = df[treatment_col].astype(str)

    return df


def preprocess_data_faster(data, treatment_col="segment", y_col="visit"):
    """
    Fast vectorized preprocessing for uplift modeling (compatible behavior).

    Compatibility preserved:
    - Numeric columns are kept as numeric only if they have >= 1000 distinct values
      (excluding treatment and outcome columns).
    - Other columns are treated as categorical and encoded by ascending uplift rank.
    - Missing categorical values are set to "NAN_VAL" then encoded as -1.
    - treatment_col is cast to string at the end.

    Bug fix included:
    - Control probability uses t0y1 / (t0y0 + t0y1), not t0y1 / (t0y1 + t0y1).
    """
    # Basic input validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame")
    if treatment_col not in data.columns:
        raise ValueError(f"`treatment_col`='{treatment_col}' not found in dataframe")
    if y_col not in data.columns:
        raise ValueError(f"`y_col`='{y_col}' not found in dataframe")

    # Work on a copy to avoid mutating caller data
    df = data.copy()

    # ---------------------------------------------------------------------
    # 1) Numeric columns selection (same logic as original function)
    # ---------------------------------------------------------------------
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if treatment_col in num_cols:
        num_cols.remove(treatment_col)
    if y_col in num_cols:
        num_cols.remove(y_col)

    kept_numeric_cols = []
    for col in num_cols:
        # Keep only high-cardinality numeric columns
        if df[col].nunique(dropna=True) >= 1000:
            col_min = df[col].min(skipna=True)
            fill_value = -1 if pd.isna(col_min) else (col_min - 1)
            df[col] = df[col].fillna(fill_value)
            kept_numeric_cols.append(col)

    # ---------------------------------------------------------------------
    # 2) Categorical columns = all remaining cols except treatment/outcome
    # ---------------------------------------------------------------------
    categorical_cols = [c for c in all_cols if c not in kept_numeric_cols and c not in {treatment_col, y_col}]

    # ---------------------------------------------------------------------
    # 3) Vectorized uplift-based encoding for each categorical column
    # ---------------------------------------------------------------------
    for cat_col in categorical_cols:
        # Preserve original behavior: missing -> "NAN_VAL"
        s = df[cat_col].astype("object").fillna("NAN_VAL")

        # Build a compact working frame
        tmp = pd.DataFrame({
            cat_col: s,
            treatment_col: df[treatment_col],
            y_col: df[y_col]
        })

        # Exclude "NAN_VAL" from uplift ranking (same as original logic)
        tmp_valid = tmp[tmp[cat_col] != "NAN_VAL"].copy()

        if tmp_valid.empty:
            # If only missing values, encode all as -1
            df[cat_col] = -1
            continue

        # Build indicator columns (int8 to reduce memory and speed up sums)
        t = tmp_valid[treatment_col]
        y = tmp_valid[y_col]

        tmp_valid["_t0y0"] = ((t == 0) & (y == 0)).astype(np.int8)
        tmp_valid["_t0y1"] = ((t == 0) & (y == 1)).astype(np.int8)
        tmp_valid["_t1y0"] = ((t == 1) & (y == 0)).astype(np.int8)
        tmp_valid["_t1y1"] = ((t == 1) & (y == 1)).astype(np.int8)

        # Aggregate all counts in one grouped sum (faster than groupby.apply)
        counts = (
            tmp_valid
            .groupby(cat_col, sort=False)[["_t0y0", "_t0y1", "_t1y0", "_t1y1"]]
            .sum()
            .rename(columns={
                "_t0y0": "t0y0",
                "_t0y1": "t0y1",
                "_t1y0": "t1y0",
                "_t1y1": "t1y1"
            })
        )

        # Compute denominators
        den_t = counts["t1y1"] + counts["t1y0"]
        den_c = counts["t0y1"] + counts["t0y0"]  # fixed denominator

        # Safe probabilities
        p_t = counts["t1y1"] / den_t.replace(0, np.nan)
        p_c = counts["t0y1"] / den_c.replace(0, np.nan)

        # Uplift with compatibility fallbacks
        uplift = p_t - p_c
        uplift = uplift.where(~p_t.isna(), -1.0)  # no treated samples in slice
        uplift = uplift.where(~p_c.isna(), 0.0)   # no control samples in slice

        # Stable ordering: uplift asc, then key asc as string
        order_df = pd.DataFrame({"uplift": uplift})
        order_df["_key"] = order_df.index.astype(str)
        order_df = order_df.sort_values(["uplift", "_key"], ascending=[True, True])

        # Map category value -> ordinal code
        value_to_code = {val: i for i, val in enumerate(order_df.index.tolist())}

        # Encode column; keep NAN_VAL as -1
        df[cat_col] = s.map(value_to_code).fillna(-1).astype(np.int32)

    # ---------------------------------------------------------------------
    # 4) Final compatibility cast (same as original)
    # ---------------------------------------------------------------------
    df[treatment_col] = df[treatment_col].astype(str)

    return df
