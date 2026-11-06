import pandas as pd
from pathlib import Path

RAW = Path("data/raw/hotel_bookings.csv")
CLEAN = Path("data/clean/hotel_bookings_clean.csv")

CATEGORICAL_COLS = ["hotel", "meal", "country", "market_segment", "distribution_channel",
                    "reserved_room_type", "assigned_room_type", "deposit_type",
                    "customer_type", "reservation_status"]


def examine(df: pd.DataFrame) -> None:
    """Understand the data: columns, types, ranges, groups, nulls, duplicates, misspellings."""
    print("=== Shape ===\n", df.shape)
    print("\n=== First rows ===\n", df.head())
    print("\n=== Last rows ===\n", df.tail())
    print("\n=== Columns and types ===")
    df.info()

    print("\n=== Numeric columns: min / max / mean ===")
    print(df.describe().T[["min", "max", "mean"]])

    print("\n=== Null values (count and %) ===")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    print(pd.DataFrame({"missing": nulls, "percent": (nulls / len(df) * 100).round(2)}))

    print("\n=== Duplicate rows ===\n", df.duplicated().sum())

    print("\n=== Groups (values of categorical columns) ===")
    for col in CATEGORICAL_COLS:
        print(f"\n{col} ({df[col].nunique()} values):")
        print(df[col].value_counts(dropna=False).head(15))

    print("\n=== Possible misspellings (spaces/case variants) ===")
    for col in CATEGORICAL_COLS:
        raw_n = df[col].nunique()
        norm_n = df[col].astype(str).str.strip().str.lower().nunique()
        if raw_n != norm_n:
            print(f"{col}: {raw_n} raw values vs {norm_n} after normalising -> check!")

    print("\n=== Suspicious records ===")
    print("Zero guests:", ((df["adults"] + df["children"].fillna(0) + df["babies"]) == 0).sum())
    print("Zero-night stays:", ((df["stays_in_weekend_nights"] + df["stays_in_week_nights"]) == 0).sum())
    print("Negative ADR:", (df["adr"] < 0).sum())
    print("ADR above 1000:", (df["adr"] > 1000).sum())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start = len(df)

    # Column names and text values: remove leading/trailing spaces
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Missing values (handled per column, not a blanket dropna)
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0)       # 0 = no agent
    df["company"] = df["company"].fillna(0)   # 0 = no company

    # Inconsistent labels
    df["meal"] = df["meal"].replace({"Undefined": "SC"})
    df["market_segment"] = df["market_segment"].replace({"Undefined": "Other"})
    df["distribution_channel"] = df["distribution_channel"].replace({"Undefined": "Other"})
    df["country"] = df["country"].replace({"CN": "CHN"})   # verify with examine()

    # Invalid records
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    df = df[(df["stays_in_weekend_nights"] + df["stays_in_week_nights"]) > 0]
    df = df[df["adr"] >= 0]
    df = df[df["adr"] < df["adr"].quantile(0.999)]   # extreme outliers (e.g. 5400)

    # Types
    df[["children", "agent", "company"]] = df[["children", "agent", "company"]].astype(int)
    df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])

    # Duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"\nCleaning removed {start - len(df)} rows ({start} -> {len(df)})")
    return df


if __name__ == "__main__":
    raw = pd.read_csv(RAW)
    examine(raw)
    cleaned = clean(raw)
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN, index=False)
    print(f"Saved cleaned data to {CLEAN}")