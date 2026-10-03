"""Further independent challenges to GR-PD-SAT-v1.0.

Run from the workspace root after installing requirements-statistics.txt.
Exploratory test p-values are descriptive: the supplied series are mock data,
the sample is short, and multiple specifications are deliberately challenged.
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
    het_goldfeldquandt,
    linear_rainbow,
    normal_ad,
)
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import zivot_andrews


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "pd_model_submission/data/german_retail_pd_mock.csv"
OUT = Path(__file__).with_name("expanded_results.json")
FEATURES = ["GDP_Index", "Unemployment_Rate", "ECB_Rate", "Euribor_3M"]
NAMES = ["intercept", *FEATURES]


def compact(v):
    if isinstance(v, dict):
        return {str(k): compact(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [compact(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return v


def matrix(frame, features=FEATURES):
    return sm.add_constant(frame[features].to_numpy(float), has_constant="add")


def fit(frame, features=FEATURES):
    return sm.OLS(frame.PD_Retail.to_numpy(float), matrix(frame, features)).fit()


def metric(y, p):
    d = np.asarray(p) - np.asarray(y)
    return {"n": len(d), "rmse": np.sqrt(np.mean(d*d)),
            "mae": np.mean(np.abs(d)), "bias": np.mean(d),
            "negative_predictions": int(np.sum(np.asarray(p) < 0))}


def loss_challenge(y, a, b, power=2, maxlags=3):
    """Positive difference means candidate a has more loss than comparator b."""
    ya, aa, bb = map(np.asarray, (y, a, b))
    ea = np.abs(ya-aa)**power
    eb = np.abs(ya-bb)**power
    d = ea-eb
    h = sm.OLS(d, np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {"mean_loss_difference": np.mean(d), "hac_se": h.bse[0],
            "hac_t": h.tvalues[0], "hac_p_two_sided": h.pvalues[0],
            "hac_95pct_ci": h.conf_int()[0].tolist(), "lag": maxlags,
            "a_better_count": int(np.sum(ea < eb)), "b_better_count": int(np.sum(eb < ea))}


def chow(frame, boundary):
    left = frame[frame.Date < boundary]
    right = frame[frame.Date >= boundary]
    k = 1 + len(FEATURES)
    if min(len(left), len(right)) <= k:
        return {"not_tested": "too few observations on a side"}
    pooled, fl, fr = fit(frame), fit(left), fit(right)
    rss0 = np.sum(pooled.resid**2)
    rss1 = np.sum(fl.resid**2) + np.sum(fr.resid**2)
    f = max(0, ((rss0-rss1)/k)/(rss1/(len(frame)-2*k)))
    return {"n_before": len(left), "n_after": len(right), "f": f,
            "df1": k, "df2": len(frame)-2*k,
            "p": stats.f.sf(f, k, len(frame)-2*k)}


def trimmed_mean(x, proportion=.1):
    return stats.trim_mean(np.asarray(x), proportion)


def main():
    raw = pd.read_csv(SOURCE)
    raw["Date"] = pd.to_datetime(raw.Date, errors="coerce")
    raw = raw.sort_values("Date").reset_index(drop=True)
    cc = raw.dropna(subset=["PD_Retail", *FEATURES]).reset_index(drop=True)
    full = fit(cc)
    infl = OLSInfluence(full)
    cook = infl.cooks_distance[0]
    ext = infl.resid_studentized_external
    dfbetas = infl.dfbetas
    event_pos = int(np.where(cc.Date == pd.Timestamp("2015-06-30"))[0][0])

    # Influence and parameter fragility. Deletion is a diagnostic, not data repair.
    loo = np.array([fit(cc.drop(index=i)).params for i in range(len(cc))])
    base = full.params
    sign_flip_count = {name: int(np.sum(np.sign(loo[:,j]) != np.sign(base[j])))
                       for j, name in enumerate(NAMES)}
    standardized = (cc[FEATURES]-cc[FEATURES].mean())/cc[FEATURES].std(ddof=0)
    robust_x = sm.add_constant(standardized.to_numpy(float), has_constant="add")
    huber = sm.RLM(cc.PD_Retail.to_numpy(float), robust_x, M=sm.robust.norms.HuberT()).fit()
    huber_beta = np.r_[huber.params[0]-np.sum(huber.params[1:]*cc[FEATURES].mean().to_numpy()/cc[FEATURES].std(ddof=0).to_numpy()),
                       huber.params[1:]/cc[FEATURES].std(ddof=0).to_numpy()]
    rolling = []
    for start in range(len(cc)-39):
        chunk = cc.iloc[start:start+40]
        rolling.append(fit(chunk).params)
    rolling = np.asarray(rolling)

    influence = {
        "n": len(cc), "cooks_4_over_n": 4/len(cc),
        "top_cooks": [{"date": cc.Date.iloc[i], "cook": cook[i],
                       "externally_studentized_residual": ext[i],
                       "leverage": infl.hat_matrix_diag[i]}
                      for i in np.argsort(cook)[-5:][::-1]],
        "2015q2_external_studentized_residual": ext[event_pos],
        "2015q2_dfbetas": dict(zip(NAMES, dfbetas[event_pos])),
        "max_abs_dfbetas": dict(zip(NAMES, np.max(np.abs(dfbetas), axis=0))),
        "leave_one_out_sign_flips": sign_flip_count,
        "leave_one_out_coefficient_ranges": {name: [np.min(loo[:,j]), np.max(loo[:,j])]
                                             for j,name in enumerate(NAMES)},
        "huber_coefficients": dict(zip(NAMES, huber_beta)),
        "huber_event_weight": huber.weights[event_pos],
        "rolling_40_observation_windows": len(rolling),
        "rolling_positive_share": {name: np.mean(rolling[:,j] > 0)
                                   for j,name in enumerate(NAMES)},
        "rolling_coefficient_ranges": {name: [np.min(rolling[:,j]), np.max(rolling[:,j])]
                                       for j,name in enumerate(NAMES)},
    }

    # Complement existing stationarity and residual diagnostics. These tests
    # are exploratory and evaluated with the observations in date order.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ad_stat, ad_p = normal_ad(full.resid)
        rain_stat, rain_p = linear_rainbow(full, frac=.5)
        gq_f, gq_p, gq_alt = het_goldfeldquandt(
            cc.PD_Retail.to_numpy(float), matrix(cc), split=.4, drop=.2,
            alternative="two-sided", idx=None,
        )
        cc_without = cc[cc.Date != pd.Timestamp("2015-06-30")]
        gq_without = het_goldfeldquandt(
            cc_without.PD_Retail.to_numpy(float), matrix(cc_without),
            split=.4, drop=.2, alternative="two-sided", idx=None,
        )
        za = {}
        evenly_spaced = raw.copy()
        evenly_spaced.Euribor_3M = evenly_spaced.Euribor_3M.interpolate()
        for col in ["PD_Retail", *FEATURES]:
            out = zivot_andrews(evenly_spaced[col].to_numpy(float),
                                maxlag=4, regression="ct", autolag="AIC")
            za[col] = {"stat": out[0], "p": out[1], "selected_lag": out[3],
                       "break_index": out[4], "break_date": raw.Date.iloc[out[4]]}
    # Quarter tests are on the dependent variable, not the fitted errors.
    qgroups = [raw.loc[raw.Date.dt.quarter == j,"PD_Retail"].to_numpy(float) for j in range(1,5)]
    no_event = raw[raw.Date != pd.Timestamp("2015-06-30")]
    qgroups_no = [no_event.loc[no_event.Date.dt.quarter == j,"PD_Retail"].to_numpy(float) for j in range(1,5)]
    period1 = full.resid[cc.Date < pd.Timestamp("2015-01-01")]
    period2 = full.resid[cc.Date >= pd.Timestamp("2015-01-01")]
    chow_boundaries = ["2009-01-01","2015-06-30","2020-01-01"]
    chow_all = {d: chow(cc,d) for d in chow_boundaries}
    _, chow_holm, _, _ = multipletests([chow_all[d]["p"] for d in chow_boundaries],method="holm")
    additional_tests = {
        "anderson_darling_residual_normality": {"stat": ad_stat, "p": ad_p},
        "rainbow_linearity_time_order": {"stat": rain_stat, "p": rain_p},
        "goldfeld_quandt_heteroskedasticity_time_order": {"f": gq_f, "p": gq_p, "alternative": gq_alt},
        "goldfeld_quandt_without_2015q2_diagnostic_only": {"f":gq_without[0],"p":gq_without[1]},
        "levene_median_residual_2005_14_vs_2015_24": dict(zip(["stat","p"],stats.levene(period1,period2,center="median"))),
        "quarter_anova_target": dict(zip(["stat","p"],stats.f_oneway(*qgroups))),
        "quarter_kruskal_target": dict(zip(["stat","p"],stats.kruskal(*qgroups))),
        "quarter_kruskal_target_without_2015q2": dict(zip(["stat","p"],stats.kruskal(*qgroups_no))),
        "chow_breaks":chow_all,
        "chow_holm_adjusted_p_three_boundaries":dict(zip(chow_boundaries,chow_holm)),
        "chow_breaks_without_2015q2_diagnostic_only": {
            d: chow(cc[cc.Date != pd.Timestamp("2015-06-30")],d)
            for d in ["2009-01-01","2020-01-01"]
        },
        "zivot_andrews_trend_break": za,
        "test_warnings": [str(w.message) for w in caught],
    }

    # Fixed 2020-2024 test. Same-quarter macro models are conditional checks;
    # even the lagged-macro version has no publication-vintage guarantee.
    train = cc[cc.Date < "2020-01-01"]
    test = cc[cc.Date >= "2020-01-01"]
    y = test.PD_Retail.to_numpy(float)
    p_full = fit(train).predict(matrix(test))
    p_mean = np.repeat(train.PD_Retail.mean(), len(test))
    p_ecb = fit(train,["GDP_Index","Unemployment_Rate","ECB_Rate"]).predict(matrix(test,["GDP_Index","Unemployment_Rate","ECB_Rate"]))
    p_eur = fit(train,["GDP_Index","Unemployment_Rate","Euribor_3M"]).predict(matrix(test,["GDP_Index","Unemployment_Rate","Euribor_3M"]))
    lagged = raw.copy()
    for col in FEATURES:
        lagged[col] = lagged[col].shift(1)
    lagged = lagged.dropna(subset=["PD_Retail",*FEATURES]).reset_index(drop=True)
    lag_train = lagged[lagged.Date < "2020-01-01"]
    lag_test = lagged[lagged.Date >= "2020-01-01"]
    assert test.Date.tolist() == lag_test.Date.tolist()
    p_lagmacro = fit(lag_train).predict(matrix(lag_test))
    # AR(1) forecast uses only the most recent realised dependent variable.
    ar = raw[["Date","PD_Retail"]].copy()
    ar["lag_pd"] = ar.PD_Retail.shift(1)
    ar = ar.dropna()
    ar_train = ar[ar.Date < "2020-01-01"]
    ar_test = ar[ar.Date >= "2020-01-01"]
    ar_fit = sm.OLS(ar_train.PD_Retail, sm.add_constant(ar_train[["lag_pd"]])).fit()
    p_ar = ar_fit.predict(sm.add_constant(ar_test[["lag_pd"]])).to_numpy()
    assert test.Date.tolist() == ar_test.Date.tolist()
    logit_y = np.log(train.PD_Retail.to_numpy(float)/(1-train.PD_Retail.to_numpy(float)))
    logit_fit = sm.OLS(logit_y,matrix(train)).fit()
    logit_linear = logit_fit.predict(matrix(test))
    p_logit = 1/(1+np.exp(-logit_linear))
    train_center = train[FEATURES].mean().to_numpy(float)
    train_scale = train[FEATURES].std(ddof=0).to_numpy(float)
    hx_train = sm.add_constant((train[FEATURES].to_numpy(float)-train_center)/train_scale)
    hx_test = sm.add_constant((test[FEATURES].to_numpy(float)-train_center)/train_scale)
    hfit = sm.RLM(train.PD_Retail.to_numpy(float),hx_train,M=sm.robust.norms.HuberT()).fit()
    p_huber = hfit.predict(hx_test)
    pred_interval = fit(train).get_prediction(matrix(test)).summary_frame(alpha=.05)
    coverage = np.mean((y >= pred_interval.obs_ci_lower.to_numpy()) &
                       (y <= pred_interval.obs_ci_upper.to_numpy()))
    variants = {"full_contemporaneous": p_full,
                "full_contemporaneous_floored":np.maximum(p_full,.0001),
                "training_mean": p_mean,
                "omit_euribor": p_ecb, "omit_ecb": p_eur,
                "lagged_macro": p_lagmacro, "ar1_target": p_ar,
                "logit_rate_challenger":p_logit,"huber_challenger":p_huber}
    forecast = {
        "window": [test.Date.min(),test.Date.max()],
        "train_n": len(train), "test_n": len(test),
        "metrics": {name: metric(y,p) for name,p in variants.items()},
        "full_vs_mean_squared_loss_hac": loss_challenge(y,p_full,p_mean),
        "full_vs_mean_absolute_loss_hac": loss_challenge(y,p_full,p_mean,power=1),
        "full_oos_r2_vs_mean": 1-np.sum((y-p_full)**2)/np.sum((y-p_mean)**2),
        "full_vs_lagged_macro_squared_loss_hac": loss_challenge(y,p_full,p_lagmacro),
        "full_classical_95pct_observation_interval_coverage": coverage,
        "full_classical_95pct_observation_interval_below_count": int(np.sum(y < pred_interval.obs_ci_lower.to_numpy())),
        "full_classical_95pct_observation_interval_above_count": int(np.sum(y > pred_interval.obs_ci_upper.to_numpy())),
        "ar1_train_coefficient": ar_fit.params.to_dict(),
    }

    # Refit fixed holdouts at two dates to expose window selection sensitivity.
    split_checks = {}
    for start in ["2015-01-01","2020-01-01"]:
        a = cc[cc.Date < start]
        b = cc[(cc.Date >= start) & (cc.Date < pd.Timestamp(start)+pd.DateOffset(years=5))]
        if len(b):
            actual = b.PD_Retail.to_numpy(float)
            pred = fit(a).predict(matrix(b))
            mean = np.repeat(a.PD_Retail.mean(),len(b))
            split_checks[start] = {"n_train":len(a),"n_test":len(b),
                                   "ols":metric(actual,pred),"mean":metric(actual,mean)}
    forecast["fixed_five_year_windows"] = split_checks

    # Test-window feature drift and scenario marginal-range challenge.
    drift = {}
    for col in FEATURES:
        a = train[col].to_numpy(float)
        b = test[col].to_numpy(float)
        ks = stats.ks_2samp(a,b,method="auto")
        drift[col] = {"train_mean":np.mean(a),"test_mean":np.mean(b),
                      "standardized_mean_shift":(np.mean(b)-np.mean(a))/np.std(a,ddof=1),
                      "train_min":np.min(a),"train_max":np.max(a),
                      "test_outside_train_range_count":int(np.sum((b<np.min(a))|(b>np.max(a)))),
                      "ks_stat":ks.statistic,"ks_p_exploratory":ks.pvalue}
    scenario_rows = [
        ("baseline",2750,.052,.005,.005),
        ("mild",2650,.065,.012,.013),
        ("severe",2450,.095,0,0),
        ("adverse",2520,.078,0,.003),
        ("expansion",3100,.03,.025,.025),
        ("boom",3300,.02,.035,.04),
    ]
    scenario_domain = {}
    invcov = np.linalg.pinv(np.cov(train[FEATURES].to_numpy(float),rowvar=False))
    center = train[FEATURES].mean().to_numpy(float)
    train_md2 = np.array([(v-center)@invcov@(v-center) for v in train[FEATURES].to_numpy(float)])
    for name,*values in scenario_rows:
        outside = [col for col,value in zip(FEATURES,values)
                   if value < train[col].min() or value > train[col].max()]
        v = np.asarray(values)
        md2 = (v-center)@invcov@(v-center)
        scenario_domain[name] = {"outside_train_marginal_ranges":outside,
                                 "mahalanobis_d2_train_covariance":md2,
                                 "above_max_observed_train_d2":bool(md2 > max(train_md2))}

    # A numerical audit of method claims; check all published summary quantiles
    # under the standard linear interpolation convention.
    published_quartiles = {
        "PD_Retail":[.022300,.024862,.028500],
        "GDP_Index":[2635.780,2707.290,2761.540],
        "Unemployment_Rate":[.048200,.051400,.057000],
        "ECB_Rate":[.000100,.002900,.005300],
        "Euribor_3M":[.001400,.003600,.006100],
    }
    quantiles = {}
    for col,claimed in published_quartiles.items():
        observed = raw[col].quantile([.25,.5,.75]).to_numpy(float)
        quantiles[col] = {"claimed":claimed,"computed":observed,
                          "absolute_discrepancy":np.abs(observed-claimed)}

    result = {
        "versions":{"numpy":np.__version__,"pandas":pd.__version__,
                    "scipy":scipy.__version__,"statsmodels":statsmodels.__version__},
        "data_audit":{"published_quantiles":quantiles,
                      "pd_mean":raw.PD_Retail.mean(),
                      "pd_median":raw.PD_Retail.median(),
                      "pd_10pct_trimmed_mean":trimmed_mean(raw.PD_Retail),
                      "pd_mean_without_2015q2":no_event.PD_Retail.mean(),
                      "number_of_zeros_ecb_2022_2024":int(np.sum(raw.loc[raw.Date >= "2022-01-01","ECB_Rate"] == 0))},
        "influence":influence,
        "additional_statistical_tests":additional_tests,
        "forecast_challenge":forecast,
        "feature_drift_train_2005_2019_vs_test_2020_2024":drift,
        "scenario_domain_vs_2005_2019_train":scenario_domain,
        "interpretation_limits":[
            "All p-values are exploratory; many correlated tests were examined without pre-registration.",
            "Complete-case deletion creates two gaps in an otherwise quarterly sample.",
            "The target is an unverified aggregate rate without default counts or exposure denominators.",
            "Contemporaneous forecasts condition on observed same-quarter macro values.",
            "The 2015 Q2 event is not removed or replaced in primary results.",
        ],
    }
    OUT.write_text(json.dumps(compact(result),indent=2),encoding="utf-8")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
