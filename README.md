# PD Model Validation

A probability-of-default model validation exercise using 80 quarters of synthetic data. The fictional bank submission is deliberately flawed, with separate scripts to find implementation and modelling problems.

## Run the checks

Use Python 3.12 in a virtual environment.

```sh
python -m pip install -r requirements.lock.txt
python validation/independent_validation.py
python validation/implementation_checks.py
python validation/statistical_diagnostics.py
python validation/expanded_diagnostics.py
```

Results are written to JSON files in `validation/`.

The submitted model, data and methodology are in `pd_model_submission/`. `generate_pd_data.py` recreates the synthetic CSV; rerun the checks if you replace it.

[MIT license](LICENSE).
