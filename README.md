# PD Model Validation Exercise

A small, reproducible exercise in validating a probability-of-default model. It combines a deliberately flawed model submission with independent implementation and statistical checks.

The 80 quarterly observations are synthetic. No customer records or bank data are included. References to a bank or SR 11-7 in the submission are part of the fictional exercise; the methodology document is the object being reviewed, not an endorsed validation standard.

## Run

Use Python 3.12 in a virtual environment.

```sh
python -m pip install -r requirements.lock.txt
python validation/independent_validation.py
python validation/implementation_checks.py
python validation/statistical_diagnostics.py
python validation/expanded_diagnostics.py
```

The scripts write JSON results into `validation/`. A completed script does not mean the submitted model passes validation: the reported findings intentionally expose modelling and implementation problems.

## Contents

- `generate_pd_data.py`: creates the synthetic exercise dataset
- `pd_model_submission/`: dataset, submitted model and methodology
- `validation/`: independent checks and their results

The generator writes within this repository. It replaces the synthetic CSV, so rerun validation after regenerating it. This is educational material, not a model approved for lending or regulatory use.

MIT licensed.

The dependency lock records the versions resolved for Python 3.12 on Windows on 2026-10-03. Other platforms may need a compatible local environment.
