"""Additional independent time-series diagnostics for GR-PD-SAT-v1.0.

Requires scipy==1.18.1, statsmodels==0.15.0 and patsy==1.0.3 in addition to
the bundled NumPy/Pandas runtime. No submitted files are modified.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    acorr_ljungbox,
    breaks_cusumolsresid,
    het_arch,
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.tsa.stattools import adfuller, kpss


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pd_model_submission/data/german_retail_pd_mock.csv"
OUT = Path(__file__).with_name("statistical_results.json")
FEATURES = ["GDP_Index", "Unemployment_Rate", "ECB_Rate", "Euribor_3M"]


def adf_result(x, regression="c", autolag="AIC", maxlag=8):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = adfuller(
            np.asarray(x, dtype=float), maxlag=maxlag,
            regression=regression, autolag=autolag,
            result_object=False,
        )
    stat, p, lags, nobs, crit = res[:5]
    icbest = res[5] if len(res) > 5 else None
    return dict(stat=stat, p=p, lags=lags, nobs=nobs,
                critical_5pct=crit["5%"], icbest=icbest,
                warnings=[str(w.message) for w in caught])


def kpss_result(x, regression="c"):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stat, p, lags, crit = kpss(np.asarray(x, dtype=float),
                                    regression=regression, nlags="auto")
    return dict(stat=stat, p=p, lags=lags,
                critical_5pct=crit["5%"],
                warnings=[str(w.message) for w in caught])


def diagnostics(frame):
    y = frame.PD_Retail.to_numpy(float)
    x = sm.add_constant(frame[FEATURES].to_numpy(float), has_constant="add")
    fit = sm.OLS(y, x).fit()
    e = fit.resid
    bg1 = acorr_breusch_godfrey(fit, nlags=1, result_object=False)
    bg4 = acorr_breusch_godfrey(fit, nlags=4, result_object=False)
    lb = acorr_ljungbox(e, lags=[1, 4, 8], return_df=True)
    bp_robust = het_breuschpagan(e, x, robust=True)
    bp_classic = het_breuschpagan(e, x, robust=False)
    white = het_white(e, x)
    arch = het_arch(e, nlags=4)
    jb = stats.jarque_bera(e)
    shapiro = stats.shapiro(e)
    reset = linear_reset(fit, power=2, test_type="fitted", use_f=True)
    cusum = breaks_cusumolsresid(e, ddof=x.shape[1])
    return {
        "n": len(frame),
        "jarque_bera": [jb.statistic, jb.pvalue],
        "shapiro": [shapiro.statistic, shapiro.pvalue],
        "breusch_pagan_studentized_lm": list(bp_robust),
        "breusch_pagan_classical_lm": list(bp_classic),
        "white_lm": list(white),
        "arch_lm_lag4": list(arch),
        "breusch_godfrey_lag1": list(bg1),
        "breusch_godfrey_lag4": list(bg4),
        "ljung_box": {str(i): {"stat": lb.loc[i, "lb_stat"],
                               "p": lb.loc[i, "lb_pvalue"]} for i in [1, 4, 8]},
        "reset_f": [float(reset.statistic), float(reset.pvalue)],
        "cusum": [cusum[0], cusum[1]],
    }


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def main():
    raw = pd.read_csv(DATA).sort_values("Date").reset_index(drop=True)
    cc = raw.dropna(subset=["PD_Retail"] + FEATURES)
    interp = raw.copy()
    interp.Euribor_3M = interp.Euribor_3M.interpolate(method="linear")
    without_event = cc[cc.Date != "2015-06-30"]
    spike_replaced = interp.copy()
    spike_idx = int(np.where(spike_replaced.Date.to_numpy() == "2015-06-30")[0][0])
    spike_replaced.loc[spike_idx, "PD_Retail"] = np.mean(
        [spike_replaced.loc[spike_idx - 1, "PD_Retail"],
         spike_replaced.loc[spike_idx + 1, "PD_Retail"]]
    )
    results = {
        "versions": {"scipy": scipy.__version__,
                     "statsmodels": statsmodels.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "stationarity": {},
        "residual_diagnostics_complete_cases": diagnostics(cc),
        "residual_diagnostics_interpolated_euribor": diagnostics(interp),
        "residual_diagnostics_without_2015q2": diagnostics(without_event),
        "residual_diagnostics_interpolated_euribor_and_2015q2_target": diagnostics(spike_replaced),
    }
    for col in ["PD_Retail"] + FEATURES:
        # Two missing Euribor observations are interpolated solely to restore
        # the equally spaced quarterly grid needed for the unit-root tests.
        z = interp[col].to_numpy(float)
        results["stationarity"][col] = {
            "adf_intercept_aic_max8": adf_result(z, "c"),
            "adf_intercept_trend_aic_max8": adf_result(z, "ct"),
            "adf_intercept_fixed1": adf_result(z, "c", autolag=None, maxlag=1),
            "adf_intercept_fixed4": adf_result(z, "c", autolag=None, maxlag=4),
            "adf_intercept_fixed_lags_0_to_8": {
                str(j): adf_result(z, "c", autolag=None, maxlag=j)
                for j in range(9)
            },
            "kpss_level_auto": kpss_result(z, "c"),
            "kpss_trend_auto": kpss_result(z, "ct"),
        }
    eur_ffill = raw.Euribor_3M.ffill().to_numpy(float)
    results["stationarity"]["Euribor_3M"]["adf_forward_fill_sensitivity"] = adf_result(eur_ffill, "c")
    results["stationarity"]["Euribor_3M"]["adf_complete_case_fixed1_diagnostic_only"] = adf_result(
        cc.Euribor_3M.to_numpy(float), "c", autolag=None, maxlag=1
    )
    for col in ["GDP_Index", "Unemployment_Rate", "PD_Retail"]:
        z = interp[col].to_numpy(float)
        results["stationarity"][col]["adf_first_difference_intercept_aic_max8"] = adf_result(np.diff(z), "c")
        results["stationarity"][col]["adf_first_difference_intercept_fixed1"] = adf_result(
            np.diff(z), "c", autolag=None, maxlag=1
        )
    pd_without = interp.loc[interp.Date != "2015-06-30", "PD_Retail"].to_numpy(float)
    # Deleting a quarter breaks spacing; this is an influence check only.
    results["stationarity"]["PD_Retail"]["adf_without_2015q2_diagnostic_only"] = adf_result(pd_without, "c")
    OUT.write_text(json.dumps(clean(results), indent=2), encoding="utf-8")
    print(f"Saved statistical diagnostics to {OUT}")


if __name__ == "__main__":
    main()
