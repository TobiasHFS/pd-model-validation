# Validation checks

These scripts review the deliberately flawed synthetic model in `../pd_model_submission/`.

Run from the repository root after installing `requirements.lock.txt`:

```sh
python validation/independent_validation.py
python validation/implementation_checks.py
python validation/statistical_diagnostics.py
python validation/expanded_diagnostics.py
```

Each script replaces its own JSON output in this directory. The results record replication, implementation edge cases, time-series diagnostics, influence checks and forecast comparisons. A successful script run is not model approval.

The main residual checks use complete cases. Interpolation and changes to the 2015 Q2 observation are labelled sensitivity exercises. Exploratory p values are not acceptance thresholds. No signed validation report or regulatory endorsement is included.
