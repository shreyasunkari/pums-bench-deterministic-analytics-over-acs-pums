from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_person(year: int, usecols=None):
    path = RAW_DATA_DIR / str(year) / "psam_p48.csv"

    kwargs = {
        "usecols": usecols,
        "low_memory": False,
    }

    if usecols is None or "SERIALNO" in usecols:
        kwargs["dtype"] = {"SERIALNO": "string"}

    return pd.read_csv(path, **kwargs)


def load_household(year: int, usecols=None):
    path = RAW_DATA_DIR / str(year) / "psam_h48.csv"

    kwargs = {
        "usecols": usecols,
        "low_memory": False,
    }

    if usecols is None or "SERIALNO" in usecols:
        kwargs["dtype"] = {"SERIALNO": "string"}

    return pd.read_csv(path, **kwargs)


if __name__ == "__main__":
    people_2024 = load_person(
        2024,
        usecols=["SERIALNO", "SPORDER", "AGEP", "PWGTP"]
    )

    households_2024 = load_household(
        2024,
        usecols=["SERIALNO", "WGTP"]
    )

    print("2024 person rows:", len(people_2024))
    print("Person columns:", people_2024.columns.tolist())

    print()
    print("2024 household rows:", len(households_2024))
    print("Household columns:", households_2024.columns.tolist())

    print()
    print(people_2024.head())
    print()
    print(households_2024.head())
