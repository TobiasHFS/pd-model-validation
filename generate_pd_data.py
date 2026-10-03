import pandas as pd
import numpy as np

np.random.seed(42)

# Generate quarterly dates from 2005 Q1 to 2024 Q4 (80 quarters)
dates = pd.date_range(start='2005-01-01', end='2024-12-31', freq='QE')
n = len(dates)

# 1. Macroeconomic Variables
# GDP: Non-stationary, underlying upward trend
gdp_base = 2500 # Billions
gdp_growth = np.random.normal(0.003, 0.005, n) # Quarterly growth
# Add COVID shock in 2020 Q1/Q2
gdp_growth[dates.year == 2020] -= 0.02
gdp_index = gdp_base * np.cumprod(1 + gdp_growth)

# Unemployment Rate (stationary but persistent AR(1))
unemp = np.zeros(n)
unemp[0] = 0.08
for i in range(1, n):
    unemp[i] = 0.05 + 0.85 * (unemp[i-1] - 0.05) + np.random.normal(0, 0.002)
# Shock during 2008
unemp[(dates.year >= 2008) & (dates.year <= 2010)] += 0.015

# Interest Rates (Euribor 3M and ECB Rate - highly collinear)
ecb_rate = np.zeros(n)
ecb_rate[0] = 0.02
for i in range(1, n):
    ecb_rate[i] = 0.9 * ecb_rate[i-1] + np.random.normal(0, 0.001)
ecb_rate = np.clip(ecb_rate, 0, 0.05)
euribor_3m = ecb_rate + np.random.normal(0.001, 0.0005, n)

# 2. True Model for PD (Logit-transformed)
# Logit(PD) = beta_0 + beta_1 * YoY_GDP_Growth + beta_2 * diff_Unemployment + AR(1) + noise
yoy_gdp_growth = pd.Series(gdp_index).pct_change(4).fillna(0).values
diff_unemp = pd.Series(unemp).diff().fillna(0).values

# Structural equation
beta_0 = -3.5  # Base log odds corresponding to ~3% PD
beta_1 = -10.0 # Higher GDP growth -> lower PD
beta_2 = 5.0   # Increasing unemployment -> higher PD

# Heteroskedasticity: variance is higher during financial crisis (2008-2009) and COVID (2020)
sigma = np.ones(n) * 0.1
sigma[(dates.year >= 2008) & (dates.year <= 2009)] = 0.3
sigma[dates.year == 2020] = 0.4

noise = np.random.normal(0, sigma, n)

logit_pd = np.zeros(n)
logit_pd[0] = beta_0
for i in range(1, n):
    logit_pd[i] = beta_0 + beta_1 * yoy_gdp_growth[i] + beta_2 * diff_unemp[i] + 0.6 * (logit_pd[i-1] - beta_0) + noise[i]

pd_rate = 1 / (1 + np.exp(-logit_pd))

# 3. Add Data Quality Issues (Missing values, outliers)
# Add an outlier to PD in 2015 Q2 (index 41)
pd_rate[41] = 0.15 
# Add some missing values to Euribor
euribor_3m[20] = np.nan
euribor_3m[55] = np.nan

df = pd.DataFrame({
    'Date': dates,
    'PD_Retail': pd_rate,
    'GDP_Index': gdp_index,
    'Unemployment_Rate': unemp,
    'ECB_Rate': ecb_rate,
    'Euribor_3M': euribor_3m
})

from pathlib import Path
output = Path(__file__).resolve().parent / 'pd_model_submission' / 'data' / 'german_retail_pd_mock.csv'
output.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output, index=False)
print("Data generation complete.")
