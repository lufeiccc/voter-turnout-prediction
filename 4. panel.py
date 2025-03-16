import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from linearmodels.panel import PanelOLS
from linearmodels.panel import RandomEffects
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = pd.read_csv('data/cleanedData.csv')

#calculate the voter turnout rate
data['voter_turnout'] = data['Total_Votes'] / data['Eligible Voters']

#calculate the population density
data['Total Population'] = data['Total Population'].replace({',': ''}, regex=True).astype(float)
data['Square Land Miles'] = data['Square Land Miles'].replace({',': ''}, regex=True).astype(float)
data['Population Density'] = data['Total Population'] / data['Square Land Miles']

x = data[['% High School', "% Bachelor's or Higher", '% Unemployment Rate', '% Income < 50000',
          '% Hispanic', '% Black', '% Native American', '% Asian', '% Hawaiian', '% Other Races',
          '% More Than One Races', '% Male', '% Female', '% Diversity Index 2020', 'Population Density', 'Prev Democratic Vote Share']]

x_with_const = sm.add_constant(x)

# Calculate VIF for each independent variable
vif = pd.DataFrame()
vif["Variable"] = x_with_const.columns
vif["VIF"] = [variance_inflation_factor(x_with_const.values, i) for i in range(x_with_const.shape[1])]

print(vif)


# Remove variables with high multicollinearity
data = data.set_index(['County', 'Year'])
x_filtered = data[['% High School', "% Bachelor's or Higher", '% Unemployment Rate', '% Income < 50000', 
                   '% Native American', '% Hawaiian', '% Other Races', '% More Than One Races', 'Prev Democratic Vote Share']]
y = data['voter_turnout']

x_filtered = sm.add_constant(x_filtered)
data_clean = pd.concat([y, x_filtered], axis=1)

# Separate the cleaned dependent and independent variables
y_clean = data_clean['voter_turnout']
x_clean = data_clean.drop(columns=['voter_turnout'])

# Run panel regression (Fixed Effects with entity and time effects)
model_fe_clean = PanelOLS(y_clean, x_clean, entity_effects=True, time_effects=True)
results_fe_clean = model_fe_clean.fit()

print("result for fixed effect model")
print(results_fe_clean)

# ----------------------------------------------------------------------------------
# Run panel regression (Random Effects with entity and time effects)
model_re = RandomEffects(y_clean, x_clean)
results_re = model_re.fit()

print("result for random effect model")
print(results_re.summary)

# to calculate BIC
# BIC =log(n)k-210g(likelihood)
# Where:
# ㆍ k is the number of parameters
# ㆍ n is the number of observations

fx_loglik = results_fe_clean._loglik
fx_likelihood = np.exp(fx_loglik)
rd_loglik = results_re._loglik
rd_likelihood = np.exp(rd_loglik)
# entity fixed effect (67-1)
# time fixed effect 1
fx_k = results_re.df_model + 66 + 1
fx_n = results_re.nobs

# entity random effect 1
# residual (error term)
rd_k = results_re.params.shape[0] + 2
rd_n = results_re.nobs

fx_bic = np.log(fx_n) * fx_k - 2 * np.log(fx_likelihood)
rd_bic = np.log(rd_n) * rd_k - 2 * np.log(rd_likelihood)

print(f"BIC for fixed effects is: {fx_bic}")
print(f"BIC for random effects is: {rd_bic}")