# Validation checks

Run these scripts from the repository root after installing `requirements.lock.txt`:

```sh
python validation/independent_validation.py
python validation/implementation_checks.py
python validation/statistical_diagnostics.py
python validation/expanded_diagnostics.py
```

Each script replaces its own JSON result file. They cover replication, implementation edge cases, time-series diagnostics, influence and forecast comparisons.

The main residual checks use complete cases. Interpolation and changes to the 2015 Q2 observation are sensitivity exercises. The p values are exploratory.
