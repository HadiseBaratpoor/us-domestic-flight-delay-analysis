"""Recompute descriptive evidence from the supplied CSV; does not train models."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COLUMNS = [
    "Unnamed: 0", "flight_date", "unique_carrier", "flight_num", "origin",
    "dest", "arr_delay", "cancelled", "distance", "carrier_delay",
    "weather_delay", "late_aircraft_delay", "nas_delay", "security_delay",
    "actual_elapsed_time",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "transport_data_2015_january.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "results")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("The CSV schema differs from the supplied project dataset.")
    dates = pd.to_datetime(df["flight_date"], errors="raise")
    observed = df["arr_delay"].notna()
    positive = df["arr_delay"].gt(0)
    if not observed.any():
        raise ValueError("No observed arrival delays are available to summarize.")

    carriers = (
        df.assign(observed_arrival=observed, delayed_arrival=positive)
        .groupby("unique_carrier")
        .agg(
            total_records=("unique_carrier", "size"),
            observed_arrivals=("observed_arrival", "sum"),
            delayed_arrivals=("delayed_arrival", "sum"),
        )
    )
    carriers["missing_arrivals"] = carriers.total_records - carriers.observed_arrivals
    carriers["delay_percent_observed"] = (
        100 * carriers.delayed_arrivals / carriers.observed_arrivals.replace(0, float("nan"))
    )
    carriers["delay_percent_all_records_legacy"] = 100 * carriers.delayed_arrivals / carriers.total_records
    numeric = df.select_dtypes(include="number").drop(columns="Unnamed: 0")
    airport_counts = df.origin.value_counts()
    busy = df[df.origin.isin(airport_counts.head(20).index)]
    summary = {
        "dataset_file": args.data.name,
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows": len(df),
        "columns": list(df.columns),
        "date_start": dates.min().date().isoformat(),
        "date_end": dates.max().date().isoformat(),
        "distinct_dates": int(dates.nunique()),
        "carriers": int(df.unique_carrier.nunique()),
        "origin_airports": int(df.origin.nunique()),
        "destination_airports": int(df.dest.nunique()),
        "directional_routes": len(df[["origin", "dest"]].drop_duplicates()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "observed_arrivals": int(observed.sum()),
        "positive_arrival_delays": int(positive.sum()),
        "early_arrivals": int(df.arr_delay.lt(0).sum()),
        "exactly_on_time_arrivals": int(df.arr_delay.eq(0).sum()),
        "delay_percent_observed": float(100 * positive.sum() / observed.sum()),
        "delay_percent_all_records_legacy": float(100 * positive.mean()),
        "cancelled_records": int(df.cancelled.eq(1).sum()),
        "missing_arrival_not_marked_cancelled": int((~observed & df.cancelled.eq(0)).sum()),
        "complete_case_records": len(df.dropna()),
        "arrival_delays_at_least_15_minutes": int(df.arr_delay.ge(15).sum()),
        "duplicates_including_index": int(df.duplicated().sum()),
        "duplicates_excluding_index": int(df.drop(columns="Unnamed: 0").duplicated().sum()),
        "index_is_zero_based_sequence": bool(df["Unnamed: 0"].eq(range(len(df))).all()),
        "numeric_summary": numeric.describe().to_dict(),
        "daily_record_counts": {k.date().isoformat(): int(v) for k, v in dates.value_counts().sort_index().items()},
        "weekday_record_counts": dates.dt.day_name().value_counts().sort_index().to_dict(),
        "weekday_delayed_counts": dates[positive].dt.day_name().value_counts().sort_index().to_dict(),
        "busiest_origins_by_record_count": airport_counts.head(20).to_dict(),
        "pearson_distance_elapsed_all_observed_pairs": float(df.distance.corr(df.actual_elapsed_time)),
        "pearson_arrival_delay_elapsed_top_20_origins": float(busy.arr_delay.corr(busy.actual_elapsed_time)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_profile.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    carriers.to_csv(args.output_dir / "carrier_summary.csv", float_format="%.6f")

    plot_data = carriers.dropna(subset=["delay_percent_observed"]).sort_values("delay_percent_observed")
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(plot_data.index, plot_data.delay_percent_observed, color="#2563a6", height=0.65)
    ax.bar_label(bars, fmt="%.2f%%", padding=5, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Arrivals with delay > 0 minutes / arrivals with known delay (%)")
    ax.set_ylabel("Carrier code")
    ax.set_title(
        f"Arrival delays by carrier | {summary['date_start']} to {summary['date_end']}",
        loc="left", fontsize=14, pad=16,
    )
    ax.xaxis.grid(True, color="#e5e7eb")
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.text(
        0.12, 0.025,
        f"{int(observed.sum()):,} known arrivals; {int((~observed).sum()):,} missing arrivals excluded. "
        "Describes this sample only.", fontsize=9, color="#374151",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(args.output_dir / "carrier_delay_rates.png", dpi=160)
    plt.close(fig)
    print(f"Records: {len(df):,}; known arrival delays: {int(observed.sum()):,}")
    print(f"Date coverage: {summary['date_start']} to {summary['date_end']}")
    print(f"Delayed among known arrivals: {summary['delay_percent_observed']:.2f}%")
    print(f"Wrote dataset_profile.json, carrier_summary.csv, and carrier_delay_rates.png to {args.output_dir}")


if __name__ == "__main__":
    main()
