from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import math


def round_half_away_from_zero(value, digits=0):
    quantum = Decimal("1").scaleb(-digits)

    return Decimal(str(value)).quantize(
        quantum,
        rounding=ROUND_HALF_UP,
    )


def weighted_quantile(values, weights, q):
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1")

    df = pd.DataFrame({
        "value": values,
        "weight": weights,
    }).dropna()

    # Aggregate equal values before computing the cumulative weight.
    # This is important for an exact definition of:
    # sum(weight for records with value <= v)
    grouped = (
        df.groupby("value", as_index=False, sort=True)["weight"]
        .sum()
    )

    total_weight = grouped["weight"].sum()

    if total_weight <= 0:
        raise ValueError("Total weight must be positive")

    threshold = q * total_weight
    cumulative_weight = grouped["weight"].cumsum()

    return grouped.loc[
        cumulative_weight >= threshold, "value"
    ].iloc[0]


def sdr_standard_error(full_estimate, replicate_estimates):
    if len(replicate_estimates) != 80:
        raise ValueError(
            f"Expected 80 replicate estimates, got {len(replicate_estimates)}"
        )

    squared_differences = [
        (replicate - full_estimate) ** 2
        for replicate in replicate_estimates
    ]

    variance = (4 / 80) * sum(squared_differences)

    return math.sqrt(variance)
