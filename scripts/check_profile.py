"""Compare regenerated data reports with the committed reference, allowing float rounding noise."""

import argparse
import csv
import json
import math
from pathlib import Path

from matplotlib.image import imread


ROOT = Path(__file__).resolve().parents[1]


def compare(actual, expected, location="profile"):
    if type(actual) is not type(expected):
        raise ValueError(f"{location}: value type changed")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{location}: keys changed")
        for key in expected:
            compare(actual[key], expected[key], f"{location}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{location}: list length changed")
        for index, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f"{location}[{index}]")
    elif isinstance(expected, float):
        if not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f"{location}: expected {expected}, got {actual}")
    elif actual != expected:
        raise ValueError(f"{location}: expected {expected!r}, got {actual!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated_dir", type=Path)
    args = parser.parse_args()
    reference = ROOT / "docs" / "results"
    actual = json.loads((args.generated_dir / "dataset_profile.json").read_text(encoding="utf-8"))
    expected = json.loads((reference / "dataset_profile.json").read_text(encoding="utf-8"))
    compare(actual, expected)
    with (args.generated_dir / "carrier_summary.csv").open(newline="", encoding="utf-8") as actual_file:
        with (reference / "carrier_summary.csv").open(newline="", encoding="utf-8") as expected_file:
            compare(list(csv.reader(actual_file)), list(csv.reader(expected_file)), "carrier_summary.csv")
    # Font rendering varies by runner; verify PNG integrity rather than byte equality.
    for directory in [reference, args.generated_dir]:
        path = directory / "carrier_delay_rates.png"
        if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{directory}: chart is not a PNG")
        chart = imread(path)
        if chart.ndim < 2 or min(chart.shape[:2]) <= 0:
            raise ValueError(f"{directory}: invalid chart dimensions")
    print("Dataset checksum, counts, statistics, and carrier table match the committed reports.")
    print("Committed and regenerated chart files are valid PNGs; pixel equality is not checked.")


if __name__ == "__main__":
    main()
