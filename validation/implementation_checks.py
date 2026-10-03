"""Behavioral checks of development code outside its happy path."""

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "pd_model_submission/src/model_development.py"
spec = importlib.util.spec_from_file_location("submitted_model", CODE)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

results = {}

for name, y, x in [
    ("singular_design", np.array([.01, .02, .03]), np.array([[1, 1], [1, 1], [1, 1]])),
    ("perfect_fit", np.array([.01, .02, .03]), np.array([[1, 1], [1, 2], [1, 3]])),
]:
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = model.estimate_ols(y, x, ["intercept", "x"])
        results[name] = {"status": "returned", "warnings": [str(z.message) for z in w], "r2": r["r2"]}
    except Exception as exc:
        results[name] = {"status": type(exc).__name__, "message": str(exc)}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    frame = pd.DataFrame({"a": [1., 1., 1.], "b": [1., 2., 3.]})
    results["constant_vif"] = {
        "output": model.compute_vifs(frame, ["a", "b"]),
        "warnings": [str(z.message) for z in w],
    }

# The scenario routine has a lower floor but no upper bound.
scenarios = model.run_scenario_stress_testing(np.array([1.1, 0, 0, 0, 0]))
results["pd_above_one"] = {
    "first_raw": scenarios.loc[0, "Raw_Predicted_PD"],
    "first_final": scenarios.loc[0, "Final_Bounded_PD"],
}

out = Path(__file__).with_name("implementation_results.json")
out.write_text(json.dumps(results, indent=2, allow_nan=True), encoding="utf-8")
print(json.dumps(results, indent=2, allow_nan=True))
