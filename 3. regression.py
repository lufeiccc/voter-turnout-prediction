import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats

# ordinal linear regression
# --------------------------------------------------------------------------
data = pd.read_csv('data/cleanedData.csv')

# calculate voter turnout as dependent
turnout_2020 = []
turnout_2016 = []
turnout_all = []

for idx, item in enumerate(data['Total_Votes']):
    if data.loc[idx, 'Year'] == 2020:
        turnout_2020.append(
            (item / data.loc[idx, 'Eligible Voters'])
        )

for idx, item in enumerate(data['Total_Votes']):
    if data.loc[idx, 'Year'] == 2016:
        turnout_2016.append(
            (item / data.loc[idx, 'Eligible Voters'])
        )

for idx, item in enumerate(data['Total_Votes']):
    turnout_all.append(
        (item / data.loc[idx, 'Eligible Voters'])
    )

# calculate population density as independent
density_2020 = []
density_2016 = []
density_all = []

for idx, item in enumerate(data['Square Land Miles']):
    if data.loc[idx, 'Year'] == 2020:
        density_2020.append(
            (float(''.join(data.loc[idx, 'Total Population'].split(','))) / 
             float(''.join(item.split(','))))
        )

for idx, item in enumerate(data['Square Land Miles']):
    if data.loc[idx, 'Year'] == 2016:
        density_2016.append(
            (float(''.join(data.loc[idx, 'Total Population'].split(','))) / 
             float(''.join(item.split(','))))
        )

for idx, item in enumerate(data['Square Land Miles']):
    density_all.append(
        (float(''.join(data.loc[idx, 'Total Population'].split(','))) / 
            float(''.join(item.split(','))))
    )

# calculate a year variable
# it is a dummy variable where 1 represents 2020 and 0 represent 2016
year_all = []

for idx, item in enumerate(data['Year']):
    if item == 2020:
        year_all.append(1)
    else:
        year_all.append(0)


# the functions that put data frame into list
"""
Converts a specific year's data from a DataFrame column into a list of floats.

Parameters:
- name: The name of the column to extract data from.
- year: The specific year to filter the data.
"""
def to_list(name, year):
    temp = []
    for idx, item in enumerate(data[name]):
        if data.loc[idx, 'Year'] == year:
            temp.append(float(item))
    return temp


"""
Converts all data from a DataFrame column into a list of floats.

Parameters:
- name: The name of the column to extract data from.
"""
def to_list_all(name):
    temp = []
    for idx, item in enumerate(data[name]):
        temp.append(float(item))
    return temp

# creates independent variables with to_list function for 2020
college_2020 = to_list("% Bachelor's or Higher", 2020)
unemploy_2020 = to_list("% Unemployment Rate", 2020)
income_2020 = to_list("% Income < 50000", 2020)
hispanic_2020 = to_list("% Hispanic", 2020)
black_2020 = to_list("% Black", 2020)
asian_2020 = to_list("% Asian", 2020)
diversity_2020 = to_list("% Diversity Index 2020", 2020)
female_2020 = to_list("% Female", 2020)
demo_2020 = to_list('Prev Democratic Vote Share', 2020)

# creates independent variables with to_list function for 2016
college_2016 = to_list("% Bachelor's or Higher", 2016)
unemploy_2016 = to_list("% Unemployment Rate", 2016)
income_2016 = to_list("% Income < 50000", 2016)
hispanic_2016 = to_list("% Hispanic", 2016)
black_2016 = to_list("% Black", 2016)
asian_2016 = to_list("% Asian", 2016)
diversity_2016 = to_list("% Diversity Index 2020", 2016)
female_2016 = to_list("% Female", 2016)
demo_2016 = to_list('Prev Democratic Vote Share', 2016)

# creates independent variables with to_list function for all data (pool the data of 2020 and 2016 together)
college_all = to_list_all("% Bachelor's or Higher")
unemploy_all = to_list_all("% Unemployment Rate")
income_all = to_list_all("% Income < 50000")
hispanic_all = to_list_all("% Hispanic")
black_all = to_list_all("% Black")
asian_all = to_list_all("% Asian")
diversity_all = to_list_all("% Diversity Index 2020")
female_all = to_list_all("% Female")
demo_all = to_list_all('Prev Democratic Vote Share')

# create IV lists
variables_2020 = [college_2020, unemploy_2020, income_2020, 
              hispanic_2020, black_2020, asian_2020,
              diversity_2020, female_2020, demo_2020]

variablesName_2020 = ['college_2020', 'unemploy_2020', 'income_2020', 
              'hispanic_2020', 'black_2020', 'asian_2020',
              'diversity_2020', 'female_2020', 'demo_2020']

variables_2016 = [college_2016, unemploy_2016, income_2016, 
              hispanic_2016, black_2016, asian_2016,
              diversity_2016, female_2016, demo_2016]

variablesName_2016 = ['college_2016', 'unemploy_2016', 'income_2016', 
              'hispanic_2016', 'black_2016', 'asian_2016',
              'diversity_2016', 'female_2016', 'demo_2016']

variables_all = [college_all, unemploy_all, income_all, 
              hispanic_all, black_all, asian_all,
              diversity_all, female_all, year_all, demo_all]

variablesName_all = ['college_all', 'unemploy_all', 'income_all', 
              'hispanic_all', 'black_all', 'asian_all',
              'diversity_all', 'female_all', 'year_all', 'demo_all']


# create a function to generate a list which can fit sklearn

"""
Generates a list of lists fit for use as independent variables in sklearn models.

Parameters:
- xList: A list to store the generated lists of independent variables.
- yList: A list of dependent variable values (e.g., turnout data).
- variables: A list of lists or arrays containing the independent variables.
"""
def x_generator(xList, yList, variables):
    for idx in range(0, len(yList)):
        temp = []
        for v in variables:
            temp.append(v[idx])
        xList.append(temp)
    return xList

# apply to x_generator function to create three lists as IV
independent_2020 = x_generator([], turnout_2020, variables=variables_2020)
independent_2016 = x_generator([], turnout_2016, variables=variables_2016)
independent_all = x_generator([], turnout_all, variables=variables_all)


"""
Calculates p-values for model coefficients and their significance levels.

Parameters:
- X: The matrix of independent variables (features).
- Y: The dependent variable (target values).
- model: A fitted linear regression model.
"""
def p_calculator(X, Y, model):
    residuals = Y - model.predict(X)
    n, k = X.shape
    RSS = np.sum(residuals**2) 
    variance = RSS / (n - k - 1)

    # Compute the covariance matrix of the coefficients
    X_with_intercept = np.column_stack([np.ones(n), X])
    cov_matrix = variance * np.linalg.inv(X_with_intercept.T @ X_with_intercept)

    # Calculate t-statistics (coefficients / standard error)
    standard_errors = np.sqrt(np.diag(cov_matrix))
    t_stats = model.coef_ / standard_errors[1:]

    # Calculate p-values
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k - 1))

    # demonstrate the significant levels
    sigStar = []
    for item in p_values:
        if item < 0.001:
            sigStar.append('***')
        elif item < 0.01:
            sigStar.append('**')
        elif item < 0.05:
            sigStar.append('*')
        else:
            sigStar.append('')

    return p_values, sigStar




# create a function to use sklearn to do regression
"""
Performs linear regression, and calculates statistical measures.

Parameters:
- IV: List or array of independent variables (features).
- DV: List or array of dependent variable values (target).
- IVname: List of names corresponding to the independent variables.
"""
def regressionSKL(IV, DV, IVname):
    X = np.array(IV)
    Y = np.array(DV)

    model = LinearRegression()
    model.fit(X, Y)
    Y_pred = model.predict(X)

    coefficients = model.coef_.tolist()
    intercept = model.intercept_.tolist()

    n, k = X.shape
    p_values, sigStar = p_calculator(X, Y, model)

    # calculate log-likelihood
    residuals = Y - Y_pred
    sigma_squared = np.var(residuals, ddof=1)
    log_likelihood = -n/2 * np.log(2 * np.pi * sigma_squared) - (1/(2 * sigma_squared)) * np.sum(residuals**2)
    likelihood = np.exp(log_likelihood)

    # Adjusted R-squared
    r2 = r2_score(Y, Y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    # calculate BIC
    bic = np.log(n) * k - 2 * np.log(likelihood)

    print(f"sample size: {n}")
    print(f"Intercept: {intercept:.4f}")
    print(f"{'Variable':<15}{'Coefficient':<15}{'P-value':<15}{'Significance':<10}")
    for idx, item in enumerate(coefficients):
        print(f"{IVname[idx]:<15}{item:<15.4f}{p_values[idx]:<15.4f}{sigStar[idx]:<10}")
    print(f"R2 = {r2_score(Y, Y_pred):.4f}")
    print(f"Adjusted R2 = {adj_r2:.4f}")
    print(f"Log-Likelihood: {log_likelihood:.4f}")
    print(f"BIC is: {bic:.4f}")

# use sklearn to do regression with three sets of data
print("**************for 2020 sample only, n = 67**************")
regressionSKL(independent_2020, turnout_2020, variablesName_2020)
print("")
print("**************for 2016 sample only, n = 67**************")
regressionSKL(independent_2016, turnout_2016, variablesName_2016)
print("")
print("**************pool 2020 and 2016 sample, n = 134**************")
regressionSKL(independent_all, turnout_all, variablesName_all)