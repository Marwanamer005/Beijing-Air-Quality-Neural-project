"""Data loading, cleaning and splitting for the Beijing air-quality project.

Notebooks import from here so that training and inference use exactly the
same data steps (the spec forbids duplicated preprocessing).
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
TARGET = "PM2.5"

# Last timestamp of each period (inclusive). Test = everything after VAL_END.
TRAIN_END = pd.Timestamp("2016-02-29 23:00")
VAL_END = pd.Timestamp("2016-08-31 23:00")


def load_raw(raw_dir=RAW_DIR):
    """Read the 12 station CSVs and return one table sorted by station and time.

    Adds a `datetime` column built from year/month/day/hour and drops `No`
    (a row counter with no information).
    """
    files = sorted(Path(raw_dir).glob("PRSA_Data_*.csv"))
    if len(files) != 12:
        raise FileNotFoundError(f"Expected 12 station files in {raw_dir}, found {len(files)}")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns="No")
    return df.sort_values(["station", "datetime"]).reset_index(drop=True)


def time_split(df, train_end=TRAIN_END, val_end=VAL_END):
    """Split by time, never shuffled: train <= train_end < val <= val_end < test.

    Every station is cut at the same timestamps, so no period appears in two splits.
    """
    train = df[df["datetime"] <= train_end]
    val = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)]
    test = df[df["datetime"] > val_end]
    return train.copy(), val.copy(), test.copy()


def check_no_leakage(train, val, test):
    """Raise an AssertionError if the splits overlap in time or lose/duplicate rows.

    Returns a small summary table (rows and date range per split).
    """
    assert train["datetime"].max() < val["datetime"].min(), "train overlaps validation"
    assert val["datetime"].max() < test["datetime"].min(), "validation overlaps test"

    keys = [set(zip(s["station"], s["datetime"])) for s in (train, val, test)]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]), "a row is in two splits"

    return pd.DataFrame({
        "rows": [len(s) for s in (train, val, test)],
        "from": [s["datetime"].min() for s in (train, val, test)],
        "to": [s["datetime"].max() for s in (train, val, test)],
    }, index=["train", "validation", "test"])
