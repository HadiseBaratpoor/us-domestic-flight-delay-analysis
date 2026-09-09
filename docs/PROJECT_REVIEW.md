# Project review and repository assessment

## Assessment

This is a single notebook-oriented **US domestic flight delay exploration and classification study**. The initial directory contained exactly three regular files, with no subdirectories, hidden configuration, Git repository, dependency manifest, README, test suite, or deployment assets. All three files are related. The material does not establish a commercial relationship to Sepehran Airlines, a production application, a revenue-management system, or a deployed prediction service.

The strongest supported portfolio framing is exploratory analysis and comparison of classification approaches, accompanied by an honest account of the limitations. Reliable forecasting performance is not established by this implementation.

The review inspected every original file, all 104 notebook cells and their output structures, the complete Python source, and every CSV record. Code relationships were compared using Python abstract syntax trees after excluding prose strings and comments. Original bytes were preserved.

## Original file inventory and relationships

| Original file | Size | Role | Recommendation |
| --- | ---: | --- | --- |
| `Domestic_Flights_Data_Mining.ipynb.txt` | 1,535,777 bytes | Valid notebook-format 4.0 JSON: 52 Markdown and 52 code cells, including historical tables, figures, warnings, and an error | Keep locally as the archival source; publish the identical `.ipynb` copy for notebook recognition. The `.txt` copy is now Git-ignored |
| `domestic_flights_data_mining_ipynb_txt.py` | 27,344 bytes | Closely related linear export containing analysis prose as string literals and shared code | Keep as an explicitly historical export; if reorganized later, place under `legacy/` and avoid maintaining it independently from the notebook |
| `transport_data_2015_january.csv` | 15,064,435 bytes | Input read by both analysis files; schema matches the feature selection | Keep as the shared input if redistribution is authorized; otherwise provide authorized acquisition instructions |

There are no unrelated datasets, scripts, folders, configuration files, or alternate applications in the original directory. Comments such as “for other projects,” the `Passed` / `Failure` plot labels, and an unused five-model list look like leftover generic analysis scaffolding. They do not establish that any whole file belongs to another project.

### Notebook versus Python export

Both files share the same CSV path, transformations, feature/target selection, model definitions, metrics, and unfinished ROC calls. Their active-code differences are limited to imports and tree-diagram sections:

- The notebook imports `graphviz`, `export_graphviz`, and the confusion-matrix utilities. In the script, Graphviz and confusion-matrix imports are commented out, and `export_graphviz` is absent from its tree import.
- The notebook contains active Graphviz rendering for the decision tree and one random-forest estimator. Those blocks are commented out in the script.
- The script still calls `plot_confusion_matrix`, despite commenting out its import.
- Markdown cells become standalone strings in the export. Expressions such as `df.head(2)` and `df.shape` are displayed interactively in a notebook but do not print when the script is run normally.
- The notebook retains outputs; the script contains no equivalent result snapshots. The notebook's opening dataset attribution is also absent from the export.

These differences support treating the script as a related export, not as a separate project. Exact export history and authorship cannot be reconstructed because no version history was supplied.

## Workflow map

Cell numbers below are **one-based positions**, counting both Markdown and code cells; execution counters are all unset.

| Notebook cells | Responsibility | Key connections |
| --- | --- | --- |
| 1–6 | Dataset attribution, imports, CSV loading, initial shape | Establishes the shared input |
| 7–47 | Distributions, carrier statistics, airport averages, scatter plots, and correlations | Builds `delayed`, `delayed_flights`, `proportion`, `busy_airports`, and several summaries |
| 48–55 | Plotly carrier/airport charts and weekday counts | Reuses the earlier tables and label |
| 56–65 | Model-name list and helper definitions | Custom confusion matrix, unused IQR function, bar labels, unfinished ROC helper |
| 66–77 | Missingness checks, imputation, encoding, column removal, train/test split | Mutates `df` in place before fitting |
| 78–91 | Four active classifiers and tree diagrams | All models share the same split; SVM is disabled |
| 92–97 | Metrics, confusion-matrix display, and comparison bars | Uses predictions from the four classifiers |
| 98–104 | ROC calls | Missing intermediate target/probability variables; saved error in cell 100 |

This is a sequential, stateful analysis. There are no reusable preprocessing classes, package entry points, command-line arguments in the original code, model persistence, database queries, or automated data acquisition. Re-running later cells without restarting may encounter mutated columns; for example, the date and airport fields have already been dropped. Models and derived DataFrames exist only in the process/kernel. Charts display interactively; the original source does not deliberately save a report or trained model to disk.

## Implementation details

The label is constructed at [script line 40](../domestic_flights_data_mining_ipynb_txt.py#L40), before imputation. The original rule is any strictly positive arrival delay.

At [line 487](../domestic_flights_data_mining_ipynb_txt.py#L487), three columns are dropped. The retained order is:

```text
0 Unnamed: 0          1 unique_carrier       2 flight_num
3 arr_delay           4 cancelled            5 distance
6 carrier_delay       7 weather_delay        8 late_aircraft_delay
9 nas_delay          10 security_delay      11 actual_elapsed_time
12 delayed
```

`df.values[:, 1:11]` selects positions 1 through 10, and `df.values[:, 12]` selects the label. Thus `actual_elapsed_time` is imputed and explored but is not used by the classifiers. The exported row index is not a predictor, but its presence is required for these positional slices to mean what the author intended. Removing it without rewriting selection would break the mapping.

### Active model configurations

| Model | Explicit source configuration |
| --- | --- |
| Decision tree | `criterion="entropy"`, `random_state=1`, `splitter="best"`, `max_depth=5` |
| Random forest | `RandomForestClassifier()`; all settings use installed-library defaults |
| Logistic regression | `LogisticRegression()`; all settings use installed-library defaults |
| Multilayer perceptron | `MLPClassifier(random_state=1, max_iter=100)`; remaining settings use defaults |

The split uses `test_size=0.25`, with no `random_state` or `stratify`. With the supplied 201,664 rows, that split would allocate 151,248 training rows and 50,416 test rows. Only the decision tree and MLP set estimator seeds; the split and random forest remain unseeded. No hyperparameter search, scaling, baseline, cross-validation, or external holdout is implemented.

### Imputation and encoding

Seven columns are filled with `int(column.mean())` before splitting. With the supplied data, those values are `arr_delay=13`, `carrier_delay=17`, `weather_delay=3`, `late_aircraft_delay=25`, `nas_delay=13`, `security_delay=0`, and `actual_elapsed_time=138`. Integer conversion truncates the means.

Carrier codes are mapped to integers 0–13. The code does not one-hot encode carriers or encode origins/destinations; it drops the latter. Standardization is absent. This arbitrary carrier ordering and the mixed feature scales are particularly problematic for interpreting the linear and neural-network experiments.

## Data and saved-output consistency

The supplied CSV is a 13-day sample, not a full January dataset. Its local SHA-256 and complete profile are in the [data dictionary](DATA_DICTIONARY.md) and [generated report](results/dataset_profile.json).

The saved notebook outputs cannot be treated as results of a clean run on this CSV:

| Evidence | Saved notebook | Supplied CSV / expected split |
| --- | ---: | ---: |
| Cell 6: shape | 182,940 × 15 | 201,664 × 15 |
| Cell 76: training rows | 137,205 | 151,248 |
| Cell 76: test rows | 45,735 | 50,416 |
| Cell 68: missing arrival delays | 3,765 | 5,317 |
| Cell 68: missing entries per delay-cause column | 132,136 | 143,131 |
| Cell 13: MQ total records | 362 | 12,752 |
| Cell 28: MQ known arrival delays | 316 | 11,275 |

The difference is 18,724 rows. Other saved carrier totals also differ. The source contains no documented selection step before the initial shape output that explains this. It is possible that outputs came from another dataset version or kernel state, but the precise cause is unknown. Some stored values, such as Southwest's known-arrival count of 42,020, do match the current CSV; agreement in individual cells does not resolve the broader mismatch.

All 52 code-cell execution counters are null. Cached outputs still exist, including 13 PNG and two SVG output representations, HTML tables/Plotly content, warning streams, and one error. These are historical artifacts, not proof of a successful fresh execution.

## Methodological findings

### Critical: target leakage

The classifier receives `arr_delay`, and the target is `arr_delay > 0`. Therefore the feature directly contains the labeling rule. Arrival-delay cause fields are also known after the outcome, so they are unsuitable for a pre-flight prediction task. High accuracy here primarily reflects access to outcome information; it does not demonstrate forecasting capability.

Repair requires defining the intended prediction time, selecting only information available then, and reporting new evaluation results. Merely deleting `arr_delay` while retaining all retrospective cause fields would not establish a valid pre-flight model.

### High: unknown outcomes become negative labels

Missing arrival values evaluate as false during label creation. All 5,317 unknown arrival outcomes receive class 0; 4,791 are marked cancelled, and 526 are not. The subsequent replacement of those missing arrival values with 13 leaves their labels unchanged, creating positive feature values paired with negative labels. These are not verified on-time arrivals.

Exclude unknown outcomes from supervised labeling or define a separate outcome policy before modeling. Changing the label after imputation would manufacture outcomes rather than recover them.

### High: preprocessing uses the test data

Imputation means are computed over the entire dataset before splitting. Test-set information therefore influences preprocessing. Fit data-dependent transformations only on training data, preferably in a scikit-learn pipeline. Missing delay-cause fields may reflect reporting structure, so the replacement strategy also requires domain justification.

### High: mislabeled metrics

`compute_confusion_matrix` builds rows for actual classes and columns for predicted classes. With delayed arrivals as class 1, the layout is `[[TN, FP], [FN, TP]]`.

| Source label | Actual calculation | Correct interpretation |
| --- | --- | --- |
| Accuracy | `(TN + TP) / all` | Accuracy; formula is correct |
| Classification error | `1 - accuracy` | Error rate; formula is correct |
| Sensitivity | `TN / (TN + FP)` | Specificity / recall of class 0 |
| Specificity | `TN / (TN + FN)` | Negative predictive value / precision of class 0 |

Sensitivity for delayed arrivals should be `TP / (TP + FN)`. The `Passed` and `Failure` labels in the confusion-matrix display also do not describe flight-delay classes. The custom confusion-matrix helper has no safeguards for missing classes, unexpected predicted labels, or zero denominators.

### Medium: descriptive interpretation and sampling

- **Airport count selection:** [line 303](../domestic_flights_data_mining_ipynb_txt.py#L303) calls `df.dropna()` across all columns. Only 58,533 records survive, coinciding with arrivals delayed at least 15 minutes. The resulting chart is neither total airport traffic nor all arrivals delayed by more than zero minutes.
- **Airport box plots:** the pivot at [line 208](../domestic_flights_data_mining_ipynb_txt.py#L208) averages `arr_delay` by date and origin. The box plots therefore describe distributions of daily mean delays, not individual flights. Visual impressions alone do not prove that one airport is consistently two or three times worse.
- **Weekday comparison:** Thursday appears once, while other weekdays appear twice. The code plots raw counts in alphabetical weekday order. Normalize by exposure/flight counts and use chronological weekday order for a more meaningful comparison.
- **Normal-distribution assumption:** the prose says 68.2% of flights lie within one standard deviation of the mean without establishing normality. Recomputed proportions are approximately 86.34% for MQ, 89.42% for US, and 87.07% for WN. The empirical distributions do not support the stated percentage.
- **Covariance claim:** the prose asserts a negative arrival-delay/elapsed-time relationship. For the source's top-20-origin subset, the current CSV gives Pearson correlation approximately **+0.00753**, effectively near zero. By comparison, distance and elapsed time across observed pairs have correlation approximately **+0.96264**. Scatter plots and correlation do not establish causation.
- **Carrier denominator:** the original proportions include unknown arrivals in the false class. Report known-arrival rates separately from all-record proportions and cancellations.
- **Mean summary:** calling `.value_counts()` on a one-row carrier-mean pivot counts row patterns; it does not rank carrier means. The separate `groupby(...).mean()` chart is a clearer carrier summary.
- **Chart wording:** titles such as “Airlines to avoid” and “Airports to avoid” overstate what a short, unadjusted sample demonstrates. The source also sets a y-axis label twice in the carrier-mean plot.
- **Unused statistical topics:** mode and Spearman correlation are mentioned in prose, but no mode calculation or Spearman computation is implemented. The heatmaps use the default Pearson method and repeat substantially the same analysis.

## Execution and compatibility findings

The environment used for the review is Python 3.13.15 with the direct dependencies pinned in [requirements.txt](../requirements.txt). This environment supports the added profiling script and static notebook rendering. It does not repair the historical source.

| Issue | Evidence | Suggested change for a future code revision |
| --- | --- | --- |
| Removed confusion-matrix import | Executing the original notebook import cell raises `ImportError` for `plot_confusion_matrix` in scikit-learn 1.9.0 | Import `ConfusionMatrixDisplay` and use its `from_estimator` or `from_predictions` class method |
| Non-numeric correlation input | The original `df.corr()` fails to convert date strings with pandas 3.0.5 | Explicitly choose numeric analytical columns; exclude the exported index rather than treating it as a measurement |
| Chained replacement no longer changes the parent | The source's `proportion1.unique_carrier.replace(..., inplace=True)` leaves the parent unchanged in the tested pandas release and emits a warning | Assign the replaced Series back with `proportion1["unique_carrier"] = ...` |
| Confusion-matrix function undefined in script | Import at line 20 is commented out; call at line 703 remains active | Replace both import and call consistently |
| ROC intermediates undefined | `y_onehot_test`, `class_of_interest`, `class_id`, and four probability-score arrays have no assignments; saved cell 100 contains a `NameError` for `y_onehot_test` | Compute class-aligned probabilities and a consistent positive label, or use `RocCurveDisplay.from_estimator` after repairing the model setup |
| Graphviz executable dependency | Active notebook graph cells use Python `graphviz.Source`; system `dot` was unavailable during review | Install the system Graphviz package only if tree rendering is needed; this was not verified |
| Slice assignments in weekday preparation | Historical output includes `SettingWithCopyWarning` | Start with an explicit `.copy()` of the selected rows and assign columns explicitly |

The original notebook imports are the first blocker for a clean notebook run. The Python export avoids that import but fails at `df.corr()` in a fresh execution with the tested dependencies. Fixing only those first errors would still leave later failures and the methodological problems above.

Targeted checks of the original positional Seaborn box plot, Seaborn distribution plot, and Plotly Boolean-column selection succeeded in the tested versions. They are not reported as confirmed runtime failures. More explicit argument names would still make the plotting intent clearer.

The original exact dependency versions are unknown. Saved warning paths indicate Python 3.8, and the confusion-matrix warning refers to removal in scikit-learn 1.2. This is historical context, not a complete reconstructable environment. Pinning older packages alone would not define the missing ROC variables or resolve data leakage.

Modernization references: [pandas correlation API](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html) and [scikit-learn ConfusionMatrixDisplay API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html). These references inform the proposed migrations; the failures above were also checked against the installed libraries.

## Historical model results

Cell 93 stores these scores:

| Classifier | Saved accuracy | Saved error rate |
| --- | ---: | ---: |
| Decision tree | 0.9971794031 | 0.0028205969 |
| Random forest | 0.9965015852 | 0.0034984148 |
| Logistic regression | 0.9971794031 | 0.0028205969 |
| MLP neural network | 0.9287635290 | 0.0712364710 |

They are retained for provenance only. They were not reproduced during this review, cannot be attributed confidently to the supplied CSV, and must not be presented as forecasting benchmarks. No valid ROC-AUC result is established by the supplied code/output combination. No new classifier performance is claimed.

## Incomplete, redundant, and experimental material

| Material | Classification | Recommended treatment |
| --- | --- | --- |
| SVM import, training, prediction, evaluation, and ROC references | Disabled experiment | Keep clearly labeled as disabled, or remove from a future cleaned implementation; do not advertise SVM results |
| `models` list naming five algorithms | Unused scaffolding | Update to active estimators or omit in a future revision |
| `find_outliers_IQR` | Unused helper | Keep only if an explicit, justified outlier analysis is added; do not claim outlier removal was performed |
| `plot_ROC` and final ROC calls | Incomplete implementation | Repair consistently or exclude from a future executable version |
| `BaggingClassifier`, `math`, `norm`, `classification_report`, `LabelBinarizer`, `roc_auc_score`, and other unused imports | Unused scaffolding | Remove when maintaining the source; importing them does not establish additional model/metric functionality |
| Commented Google Drive import | Colab residue | Optional historical comment; no Drive integration is active |
| Repeated correlation heatmaps | Redundant exploration | Consolidate in a future polished notebook |
| `Passed` / `Failure`, spelling mistakes, and mismatched narrative statistics | Editorial / copied scaffolding | Correct when refreshing the notebook and outputs |
| Original `.txt` plus renderable `.ipynb` | Deliberate local duplication | Publish the `.ipynb`; retain the archival `.txt` locally through the added ignore rule |
| Historical saved outputs | Useful provenance, inconsistent with current data | Preserve only with the warning, or replace after a clean, corrected rerun |

No deletion or relocation was performed. The table is a recommendation, not a record of actions already taken.

## Added documentation and reproducibility assets

- `README.md` presents the project with qualified features, setup, usage, verified descriptive findings, and limitations.
- `Domestic_Flights_Data_Mining.ipynb` is a byte-identical rendering copy, not a corrected or rerun notebook.
- `DATA_DICTIONARY.md` records the schema, observed coverage, missingness, and attribution boundary.
- `CV_PROJECT_DESCRIPTION.md` offers conservative project wording without inflated model-performance claims.
- `scripts/profile_dataset.py` computes the current descriptive summary and a static chart without changing the CSV or fitting models. It validates the expected column order, records the CSV checksum, and exposes output-path configuration.
- `docs/results/` contains newly generated descriptive results. The carrier chart excludes missing arrival outcomes and the CSV includes both denominator definitions.
- `requirements.txt` pins tested direct packages for this review. It is not a complete transitive lockfile or an original requirements file.
- `.gitignore` excludes local environments, caches, local exports, secrets files, and the redundant archival notebook extension.

## Validation boundary

Completed checks include full-file inventory, notebook JSON/schema validation, syntax compilation of all original code cells and the export, structural comparison of notebook/script logic, full-row CSV profiling with independent standard-library and pandas calculations, targeted compatibility checks, a fresh script execution to its first confirmed failure, and static HTML notebook export without execution. The added profiling workflow runs on the full supplied CSV.

The review did not complete a historical notebook run, rerun all four classifiers, render the Graphviz trees, validate browser interaction with saved Plotly output, reconstruct the historical dataset subset, or establish upstream licensing and authorship. Notebook HTML generation demonstrates local rendering, not a live GitHub preview. Nothing was published or deployed.

## Recommended publication sequence

1. Confirm authorship, original-source acknowledgments, and permission to redistribute the dataset; add an appropriate license and attribution where authorized.
2. Use the renderable `.ipynb` and these docs as an explicitly historical analytical portfolio repository. The `.txt` copy can remain local, and the original script can remain as a labeled export.
3. If the repository should promise a successful full notebook run, implement the documented runtime repairs, resolve target leakage and label policy, and regenerate every output from a clean kernel first.
4. Retain the dataset checksum and verified descriptive reports so readers can distinguish fresh results from the historical snapshot.
5. Use the CV description that matches the contributor's actual work. Avoid claims about production deployment, Sepehran business outcomes, forecasting accuracy, or completed ROC/SVM results.

The documentation is prepared for GitHub presentation, while the original analytical implementation remains historical and incomplete. A reliable runnable modeling release requires the separately identified source changes and fresh evaluation.
