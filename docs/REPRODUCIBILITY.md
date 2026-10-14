# Reproduction guide

## Environment and inputs

Run from the repository root. Install [requirements.txt](../requirements.txt) and prepare the licensed CRSP inputs listed in [README](../README.md#data-requirements).

```powershell
python -m pip install -r requirements.txt
python us_size_factor_calendar_vw.py --help
python us_size_factor_calendar_vw.py --data-dir "../CRSP数据" --output-dir "./output_calendar_rebalance_strict" --n-groups 10 --cost-bps 10 --min-listing-days 180 --fundamental-lag-days 90
```

Pass your actual data folder to `--data-dir`. The default path is now spelled consistently with the other scripts. The zip archives must have the column schemas consumed by the loader; filenames alone are insufficient.

## Three workflows

- `us_size_factor_test.py`: monthly cross-sectional study.
- `us_size_factor_test_daily.py`: daily fixed-hold study.
- `us_size_factor_calendar_vw.py`: calendar-rebalanced, value-weighted study with costs and screens.

## Outputs and comparison

The output folder receives interval/daily returns, IC, group statistics, tests, holdings and a dashboard. Root-level CSV/PNG files are selected historical artifacts and remain unchanged.

Match the data vintage, screening parameters, dates, delisting handling and cost model when comparing a rerun with the README. The original reported weak long-short spread is retained. Full backtests require external licensed data and were not run during cleanup.

## Run record

For every result you intend to compare or share, retain the repository commit, exact command/configuration, input-data identity, environment versions and output location. Save a dependency snapshot after a successful reference run; newly added requirements lists are intentionally unpinned until that validation exists.

## Validation status

Cleanup review: 14 October 2026. File paths and documented entry points were checked against the checkout. Full empirical results were not regenerated. Project-specific external prerequisites and known blockers are listed above.
