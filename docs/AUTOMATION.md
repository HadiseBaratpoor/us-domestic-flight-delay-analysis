# GitHub Actions

The repository has two workflows focused on the checks supported by its current state. They run on every branch push, pull request, and manual dispatch using GitHub-hosted Ubuntu 24.04 and Python 3.13. No branch name is hard-coded. Manual dispatch becomes available after the workflow is present on the repository's default branch.

## Repository checks

[Workflow definition](../.github/workflows/repository-checks.yml)

1. Install `requirements-ci.txt`, which includes the project requirements and a pinned Markdown parser, and check dependency compatibility.
2. Compile root-level and `scripts/` Python files without running them.
3. Validate root-level `.ipynb` files against the notebook schema and compile their code cells without executing them.
4. Check local links/images in root-level and `docs/` Markdown files, including ordinary Markdown heading anchors and source-line references. External websites are excluded from this offline check. Notebook Markdown links and custom HTML anchors are outside its scope.
5. Export the historical notebook to HTML with nbconvert, without `--execute`.
6. Upload `historical-notebook-preview` as a downloadable artifact.

The archived `.ipynb.txt` is not required by CI because it is intentionally Git-ignored. Checks operate on the published `.ipynb` copy. Saved warnings and the historical ROC error remain in the notebook; a passing structural check does not mean those cells execute successfully. The checker currently expects ordinary Python cells; adding IPython magic syntax would require updating that check.

## Dataset reproducibility

[Workflow definition](../.github/workflows/dataset-reproducibility.yml)

1. Install the project requirements and check dependency compatibility.
2. Run the full CSV profiling script into `local-exports/ci-profile/`.
3. Compare the generated JSON with `docs/results/dataset_profile.json`. Field names, types, integer counts, dates, schema, and dataset SHA-256 must agree. Floating-point statistics use relative and absolute tolerances of `1e-10` to accommodate tiny platform differences.
4. Compare the generated carrier CSV with the committed table as parsed rows, ignoring line-ending differences. Numeric values are already formatted to six decimal places by the profiling script.
5. Decode both the committed and generated PNG chart to check that they are readable. Pixel equality is not required because font rendering can vary across platforms; this is not a visual-regression check.
6. Upload `regenerated-dataset-profile`, including newly generated JSON, CSV, and PNG files. Available reports are uploaded even if the comparison fails; cancelled runs skip this upload. An absent output is an error.

The dataset must be present at its documented path. If the CSV is later excluded for redistribution reasons, update the workflow to retrieve it through an authorized, reproducible mechanism. The workflow does not download data from Kaggle or silently skip validation when the input is absent.

## Run the same checks locally

With a Python 3.13 virtual environment activated, run these commands from the repository root:

```bash
python -m pip install -r requirements-ci.txt
python -m pip check
python scripts/check_repository.py
jupyter nbconvert --to html --output-dir local-exports/notebook Domestic_Flights_Data_Mining.ipynb
python scripts/profile_dataset.py --output-dir local-exports/ci-profile
python scripts/check_profile.py local-exports/ci-profile
```

Generated local exports are Git-ignored. The validation scripts exit unsuccessfully when a check fails, so GitHub Actions marks the associated job as failed.

## Responding to failures

| Failure | What to inspect |
| --- | --- |
| Python syntax or notebook schema | The reported file/cell; preserve valid notebook JSON and Python syntax |
| Missing documentation target or anchor | The link destination, heading, filename, or source-line number |
| Dependency installation or compatibility | The package error and pinned manifests; CI does not fall back to unpinned versions |
| Dataset checksum / JSON / CSV mismatch | Whether the input, profiling logic, environment, or committed report changed; inspect the regenerated artifact |
| PNG decoding | Whether the chart is missing, truncated, or no longer a PNG |
| HTML export | Notebook format and nbconvert diagnostics; missing-alt-text warnings in historical figures do not by themselves fail the export |

When a dataset or calculation change is intentional and has been reviewed, regenerate the committed reports with `python scripts/profile_dataset.py`, inspect the chart and statistics, update affected documentation, and commit the related changes together. CI never overwrites or commits the reference reports automatically.

## Permissions, triggers, and artifacts

Both workflows use `contents: read`, disable persisted checkout credentials, have a 15-minute job timeout, and cancel superseded runs of the same workflow/event/ref. They cache pip downloads and retain artifacts for 14 days, subject to repository policy. They use ordinary `pull_request` triggers, not privileged pull-request execution. No custom secrets, API keys, cloud accounts, or deployment permissions are needed.

After pushing the workflow files, open the repository's **Actions** tab to inspect runs and download their artifacts. Nothing is deployed to GitHub Pages, no release is created, and no automatic commit is made. The HTML artifact is a historical preview, not a regenerated analytical result.

The action configuration follows the official [checkout](https://github.com/actions/checkout), [setup-python](https://github.com/actions/setup-python), [upload-artifact](https://github.com/actions/upload-artifact), and [workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax) documentation.

## Why other workflows are deferred

Full notebook execution and model evaluation should be added after repairing the runtime and methodological issues in [the technical review](PROJECT_REVIEW.md). Scheduled retraining has no current data-refresh mechanism to use. A deployment or container-publishing workflow would need an actual deployable application. These workflows are not configured for the present historical analysis repository.
