"""Independent, reproducible checks of GR-PD-SAT-v1.0.

Run from the workspace root with the bundled Python runtime. Uses NumPy and
Pandas only; it does not change the development submission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pd_model_submission/data/german_retail_pd_mock.csv"
CODE = ROOT / "pd_model_submission/src/model_development.py"
DOC = ROOT / "pd_model_submission/docs/SR11_7_PD_Model_Methodology.md"
OUT = Path(__file__).with_name("results.json")
FEATURES = ["GDP_Index", "Unemployment_Rate", "ECB_Rate", "Euribor_3M"]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    return value


def fit(frame, features=FEATURES, transform=None):
    y = frame["PD_Retail"].to_numpy(float)
    if transform == "logit":
        y = np.log(y / (1 - y))
    x = np.column_stack([np.ones(len(frame)), frame[features].to_numpy(float)])
    beta, _, rank, singular_values = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    resid = y - fitted
    return x, y, beta, fitted, resid, rank, singular_values


def metrics(actual, predicted):
    return {
        "n": len(actual),
        "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "bias_pred_minus_actual": float(np.mean(predicted - actual)),
    }


def robust_se(x, resid, lag=4):
    n, k = x.shape
    bread = np.linalg.pinv(x.T @ x)
    h = np.sum((x @ bread) * x, axis=1)
    meat = x.T @ np.diag((resid / (1 - h)) ** 2) @ x
    hc3 = np.sqrt(np.diag(bread @ meat @ bread))
    score = x * resid[:, None]
    s = score.T @ score
    for j in range(1, lag + 1):
        w = 1 - j / (lag + 1)
        a = score[j:].T @ score[:-j]
        s += w * (a + a.T)
    # Small sample correction, a specified choice rather than a claim of
    # unique HAC inference.
    hac = np.sqrt(np.diag((n / (n - k)) * bread @ s @ bread))
    return hc3, hac, h


def calibration(y, pred):
    z = np.column_stack([np.ones(len(pred)), pred])
    b = np.linalg.lstsq(z, y, rcond=None)[0]
    return {"intercept": b[0], "slope": b[1], "mean_actual": np.mean(y), "mean_pred": np.mean(pred)}


def main():
    raw = pd.read_csv(DATA)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.sort_values("Date").reset_index(drop=True)
    cc = raw.dropna(subset=["PD_Retail"] + FEATURES).copy()
    x, y, b, pred, e, rank, sv = fit(cc)
    n, k = x.shape
    rss, tss = np.sum(e * e), np.sum((y - np.mean(y)) ** 2)
    classical_se = np.sqrt(np.diag(rss / (n - k) * np.linalg.pinv(x.T @ x)))
    hc3, hac, leverage = robust_se(x, e)
    vifs = {}
    f = cc[FEATURES].to_numpy(float)
    for i, name in enumerate(FEATURES):
        others = np.column_stack([np.ones(n), np.delete(f, i, axis=1)])
        fitted_feature = others @ np.linalg.lstsq(others, f[:, i], rcond=None)[0]
        r2 = 1 - np.sum((f[:, i] - fitted_feature) ** 2) / np.sum((f[:, i] - np.mean(f[:, i])) ** 2)
        vifs[name] = 1 / (1 - r2)

    dates = raw["Date"]
    expected = pd.date_range("2005-03-31", "2024-12-31", freq="QE-DEC")
    outlier_date = pd.Timestamp("2015-06-30")
    outlier = cc[cc.Date == outlier_date]
    no_outlier = cc[cc.Date != outlier_date]
    _, _, b_no, pred_no, e_no, _, _ = fit(no_outlier)
    # Full sample fitted predictions using coefficients re-estimated without
    # the questionable observation.
    pred_no_on_full = x @ b_no
    i_out = int(np.where(cc.Date.to_numpy() == outlier_date)[0][0])
    cook = e**2 / (k * (rss / (n - k))) * leverage / (1 - leverage) ** 2
    demeaned_pd = cc.PD_Retail - cc.PD_Retail.mean()
    outlier_tss_share = float(demeaned_pd.iloc[i_out] ** 2 / np.sum(demeaned_pd**2))

    # Studentized (Koenker) auxiliary-regression variant of the BP test.
    # The submission implements the classical normal-theory BP variant.
    e2 = e**2
    e2hat = x @ np.linalg.lstsq(x, e2, rcond=None)[0]
    bp_r2 = 1 - np.sum((e2 - e2hat) ** 2) / np.sum((e2 - np.mean(e2)) ** 2)
    bp_lm = n * bp_r2

    # Pseudo out-of-time forecasts condition on observed contemporaneous
    # macro variables. They are not genuine ex-ante forecasts.
    eval_frame = cc.reset_index(drop=True)
    roll_actual, roll_full, roll_mean, roll_last, roll_logit = [], [], [], [], []
    for i in range(40, len(eval_frame)):
        train = eval_frame.iloc[:i]
        test = eval_frame.iloc[i : i + 1]
        _, _, bt, _, _, _, _ = fit(train)
        _, _, bl, _, _, _, _ = fit(train, transform="logit")
        xv = np.r_[1.0, test[FEATURES].to_numpy(float)[0]]
        roll_actual.append(float(test.PD_Retail.iloc[0]))
        roll_full.append(float(xv @ bt))
        roll_logit.append(float(1 / (1 + np.exp(-(xv @ bl)))))
        roll_mean.append(float(train.PD_Retail.mean()))
        roll_last.append(float(train.PD_Retail.iloc[-1]))
    roll_actual = np.array(roll_actual)

    hold_train = cc[cc.Date < "2020-01-01"]
    hold_test = cc[cc.Date >= "2020-01-01"]
    _, _, bh, _, _, _, _ = fit(hold_train)
    hold_x = np.column_stack([np.ones(len(hold_test)), hold_test[FEATURES].to_numpy(float)])
    hold_y = hold_test.PD_Retail.to_numpy(float)
    hold_pred = hold_x @ bh

    # Check how listwise deletion matters with transparent interpolation.
    interp = raw.copy()
    interp["Euribor_3M"] = interp.Euribor_3M.interpolate(method="linear")
    _, _, b_interp, _, _, _, _ = fit(interp)

    def sc(gdp, unemp, ecb, eur):
        return float(np.array([1, gdp, unemp, ecb, eur]) @ b)

    scenarios = {
        "baseline": sc(2750, .052, .005, .005),
        "mild": sc(2650, .065, .012, .013),
        "severe": sc(2450, .095, 0, 0),
        "adverse": sc(2520, .078, 0, .003),
        "expansion": sc(3100, .03, .025, .025),
        "boom": sc(3300, .02, .035, .04),
        "high_unemployment_increment_1pp_at_baseline": .01 * b[2],
        "rates_plus_1pp_together_at_baseline": .01 * (b[3] + b[4]),
    }

    result = {
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA, CODE, DOC]},
        "data": {
            "rows": len(raw), "columns": list(raw.columns),
            "date_min": dates.min(), "date_max": dates.max(),
            "duplicate_dates": int(dates.duplicated().sum()),
            "duplicate_rows": int(raw.duplicated().sum()),
            "missing_by_column": raw.isna().sum().to_dict(),
            "missing_quarters": [d for d in expected if d not in set(dates)],
            "invalid_dates": int(dates.isna().sum()),
            "out_of_bounds_pd": int(((raw.PD_Retail < 0) | (raw.PD_Retail > 1)).sum()),
            "negative_ecb": int((raw.ECB_Rate < 0).sum()),
            "negative_euribor": int((raw.Euribor_3M < 0).sum()),
            "min_max_mean": {c: {"min": raw[c].min(), "max": raw[c].max(), "mean": raw[c].mean(), "median": raw[c].median()} for c in ["PD_Retail"] + FEATURES},
            "2015q2": raw.loc[raw.Date == outlier_date].to_dict("records"),
            "2023q3": raw.loc[raw.Date == pd.Timestamp("2023-09-30")].to_dict("records"),
            "2024q4": raw.loc[raw.Date == pd.Timestamp("2024-12-31")].to_dict("records"),
            "ecb_zero_count_2022_2024": int((raw.loc[raw.Date >= "2022-01-01", "ECB_Rate"] == 0).sum()),
        },
        "model": {
            "n": n, "rank": rank, "coefficients": dict(zip(["intercept"] + FEATURES, b)),
            "classical_se": classical_se, "hc3_se": hc3, "hac4_se": hac,
            "r2": 1 - rss / tss, "adj_r2": 1 - (rss / (n-k)) / (tss / (n-1)),
            "f_stat": ((tss - rss) / (k-1)) / (rss / (n-k)),
            "residual_std": np.sqrt(rss / (n-k)),
            "dw": np.sum(np.diff(e)**2)/rss,
            "acf1": np.corrcoef(e[1:],e[:-1])[0,1],
            "vif": vifs, "ecb_euribor_corr": np.corrcoef(f[:,2],f[:,3])[0,1],
            "scaled_condition_number": np.linalg.cond((x[:,1:] - x[:,1:].mean(axis=0))/x[:,1:].std(axis=0)),
            "bp_lm_auxiliary": bp_lm,
            "bp_studentized_df4_pvalue": np.exp(-bp_lm/2) * (1 + bp_lm/2),
            "bp_lm_development_formula": .5 * np.sum((x @ np.linalg.lstsq(x, e2/np.mean(e2),rcond=None)[0] - 1)**2),
            "in_sample": metrics(y, pred),
            "in_sample_mean": metrics(y, np.repeat(np.mean(y), len(y))),
            "in_sample_negative_predictions": int((pred < 0).sum()),
            "in_sample_above_one_predictions": int((pred > 1).sum()),
            "calibration": calibration(y,pred),
            "outlier_tss_share": outlier_tss_share,
            "outlier_residual": e[i_out],
            "outlier_fitted": pred[i_out],
            "outlier_cook": cook[i_out],
            "cook_max": float(np.max(cook)),
            "cook_threshold_4_over_n": 4/n,
            "without_outlier_coefficients": dict(zip(["intercept"]+FEATURES,b_no)),
            "without_outlier_r2": 1 - np.sum(e_no**2)/np.sum((no_outlier.PD_Retail-no_outlier.PD_Retail.mean())**2),
            "without_outlier_rmse_rest": metrics(no_outlier.PD_Retail.to_numpy(),pred_no),
            "without_outlier_full_prediction_rmse": metrics(y,pred_no_on_full),
            "interpolation_coefficients": dict(zip(["intercept"]+FEATURES,b_interp)),
            "rolling_dates": [eval_frame.Date.iloc[40],eval_frame.Date.iloc[-1]],
            "rolling_ols": metrics(roll_actual,np.array(roll_full)),
            "rolling_ols_floored": metrics(roll_actual,np.maximum(roll_full,.0001)),
            "rolling_mean": metrics(roll_actual,np.array(roll_mean)),
            "rolling_last": metrics(roll_actual,np.array(roll_last)),
            "rolling_logit": metrics(roll_actual,np.array(roll_logit)),
            "rolling_negative_predictions": int((np.array(roll_full)<0).sum()),
            "holdout_dates": [hold_test.Date.min(), hold_test.Date.max()],
            "holdout_ols": metrics(hold_y,hold_pred),
            "holdout_train_mean": metrics(hold_y,np.repeat(hold_train.PD_Retail.mean(),len(hold_y))),
            "holdout_ols_calibration": calibration(hold_y,hold_pred),
            "holdout_negative_predictions": int((hold_pred<0).sum()),
            "scenarios": scenarios,
        },
    }
    result = clean(result)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
