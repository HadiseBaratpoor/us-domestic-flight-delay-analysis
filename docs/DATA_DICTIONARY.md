# Dataset reference

## Identity and scope

The original notebook and Python export both read `transport_data_2015_january.csv`. The file is 15,064,435 bytes and contains 201,664 data rows and 15 columns. It covers January 2–14, 2015, rather than the entire month suggested by its filename.

Verified SHA-256:

```text
054fbb03c017854cd70a0d12d2c813fd6c3a590d69d4d814de96567ef04a83bc
```

The notebook attributes the data to [this Kaggle dataset](https://www.kaggle.com/datasets/vishwanathmuthuraman/domestic-flight-data). An upstream data dictionary and license were not supplied. The descriptions below reflect the column names, stored values, and how the project uses them. They do not certify the upstream collection process.

## Schema

Types describe how the supplied CSV is read by pandas. Integer-like values such as flight numbers and cancellation flags are stored with decimal notation. String columns may use pandas' string dtype in modern releases.

| Column | Observed kind | Missing | Meaning and use |
| --- | --- | ---: | --- |
| Empty header → `Unnamed: 0` | Integer | 0 | Exported row index, exactly 0 through 201,663; not a business feature |
| `flight_date` | Date/time text | 0 | Flight date, stored with `00:00:00`; used for weekday grouping and daily airport averages, then dropped for modeling |
| `unique_carrier` | String | 0 | Two-character carrier code; used in comparisons and mapped to integers for modeling |
| `flight_num` | Numeric | 0 | Flight number; used as a numeric predictor, not a unique record identifier |
| `origin` | String | 0 | Three-letter origin airport code; used for airport summaries, then dropped for modeling |
| `dest` | String | 0 | Three-letter destination airport code; dropped for modeling |
| `arr_delay` | Numeric | 5,317 | Arrival delay in minutes as interpreted by the notebook; negative means early, zero exactly on time, positive late; used for both label construction and, incorrectly, as a predictor |
| `cancelled` | Numeric 0/1 | 0 | Cancellation indicator; 4,791 rows have value 1; included as a predictor |
| `distance` | Numeric | 0 | Route distance; used in a scatter plot and as a predictor. Units are not explicitly established by the supplied project files |
| `carrier_delay` | Numeric | 143,131 | Carrier-attributed delay duration; imputed and used as a predictor |
| `weather_delay` | Numeric | 143,131 | Weather-attributed delay duration; imputed and used as a predictor |
| `late_aircraft_delay` | Numeric | 143,131 | Late-aircraft-attributed delay duration; imputed and used as a predictor |
| `nas_delay` | Numeric | 143,131 | National Air System delay according to the notebook narrative; imputed and used as a predictor |
| `security_delay` | Numeric | 143,131 | Security-attributed delay duration; imputed and used as a predictor |
| `actual_elapsed_time` | Numeric | 5,317 | Actual elapsed flight time; explored and imputed, but excluded by the predictor slice |

The source treats the duration columns as minute-based flight measurements. Their detailed reporting rules are not supplied; missing cause values must not be assumed to mean zero without verifying those rules.

### Derived column

`delayed` is created in memory, not stored in the CSV:

```python
df["delayed"] = df["arr_delay"].apply(lambda x: x > 0)
```

The threshold is strictly greater than zero, not 15 minutes. With the original pandas loading behavior, missing arrival values compare as false and become non-delay labels. Later imputation of `arr_delay` does not update those labels.

The added profiling script preserves this definition for the count of known positive delays, but excludes missing arrival outcomes from the observed-arrival rate. It reports the original all-record rate separately.

## Completeness and coverage

| Check | Result |
| --- | ---: |
| Known arrival delays | 196,347 |
| Early arrivals | 93,552 |
| Exactly on-time arrivals | 4,168 |
| Positive arrival delays | 98,627 |
| Missing arrival and elapsed-time values | 5,317 each, on the same rows |
| Cancelled records | 4,791, all with missing arrival delay |
| Missing arrival values on rows not marked cancelled | 526 |
| Complete rows across every CSV field | 58,533 |
| Arrival delays of at least 15 minutes | 58,533 |
| Duplicate full records, including row index | 0 |
| Duplicate records after excluding row index | 0 |
| Carriers | 14 |
| Distinct origins / destinations | 312 / 312 |
| Distinct directional routes | 4,152 |

Each of the five delay-cause columns has 143,131 missing entries, approximately 70.97% of the dataset. In this file, complete cases coincide with the records whose arrival delay is at least 15 minutes. Consequently, the original airport visualization's `df.dropna()` selects this narrower delay subset, rather than all flights or all arrivals delayed by more than zero minutes. This is an observed relationship, not a verified statement of the upstream reporting policy.

The 526 records with unknown arrival time that are not marked cancelled cannot be classified further from the supplied fields alone. There is no diversion flag.

Thursday appears once; every other weekday appears twice. Raw weekday counts should not be interpreted as comparable weekday risks without normalization. There is no time-of-day resolution in `flight_date` as supplied.

## Carrier mapping in the source

The following names are copied from the source mapping and describe its historical carrier labels, not current airline operations.

| Code | Source label | Model integer |
| --- | --- | ---: |
| AA | American Airlines Inc | 0 |
| AS | Alaska Airlines Inc | 1 |
| B6 | JetBlue Airways | 2 |
| DL | Delta Air Lines Inc | 3 |
| EV | ExpressJet Airlines Inc | 4 |
| F9 | Frontier Airlines Inc | 5 |
| HA | Hawaiian Airlines Inc | 6 |
| MQ | Envoy Air | 7 |
| NK | Spirit Air Lines | 8 |
| OO | SkyWest Airlines Inc | 9 |
| UA | United Air Lines Inc | 10 |
| US | US Airways Inc | 11 |
| VX | Virgin America | 12 |
| WN | Southwest Airlines Co | 13 |

All observed carrier codes are covered by this mapping. The assigned integers are arbitrary category codes, not airline rankings.

## Recompute the reference

From the repository root, with the documented environment activated:

```bash
python scripts/profile_dataset.py
```

See [dataset_profile.json](results/dataset_profile.json) for exact numeric summaries, daily coverage, missingness, duplicate checks, and correlations; [carrier_summary.csv](results/carrier_summary.csv) contains per-carrier counts and both delay-rate denominators.
