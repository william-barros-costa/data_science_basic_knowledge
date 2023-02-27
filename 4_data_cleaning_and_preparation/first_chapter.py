# %% Imports
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

# %% Types of missing data
float_serie = Series([1.2, -3.5, np.nan, None, 0])

float_serie, float_serie.isna()

# %% Filter missing Data
data = pd.Series([1, np.nan, 3.5, np.nan, 7])

data.dropna(), data[data.notna()]

# %% Remove row with all missing data
data = pd.DataFrame([[1., 6.5, 3.], [1., np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, 6.5, 3.]])

data, data.dropna(how="all")

# %% Remove Column with all missing data
data[4] = np.nan

data, data.dropna(how="all", axis="columns")

# %% Remove rows with mostly missing data
frame = DataFrame(np.ones((7, 3)))
frame.iloc[:4, 1] = np.nan
frame.iloc[:2, 2] = np.nan

# Thresh = 2 means all rows with Count(NaN) >= 2 are removed
frame, frame.dropna(), frame.dropna(thresh=2)

# %% Fill Missing Data
frame, frame.fillna(0)

# %% Fill with different values according to column
frame, frame.fillna({1: 0.5, 2: 0})

# %% We can also fill with interpolation
frame, frame.fillna(method="ffill"), frame.fillna(data.mean())

# %% Remove duplicates
data = pd.DataFrame({"k1": ["one", "two"] * 3 + ["two"], "k2": [1, 1, 2, 3, 3, 4, 4]})
data, data.duplicated(), data.drop_duplicates()

# %% Remove duplicates by column. Usually the first stays but you can change
# this behaviour with 'keep'
data, data.drop_duplicates(subset=["k1"], keep="last")

# %% Transformation
data = pd.DataFrame({
    "food": ["bacon", "pulled pork", "bacon", "pastrami", "corned beef", "bacon", "pastrami", "honey ham", "nova lox"], 
    "ounces": [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

meat_to_animal = {
  "bacon": "pig",
  "pulled pork": "pig",
  "pastrami": "cow",
  "corned beef": "cow",
  "honey ham": "pig",
  "nova lox": "salmon"
}

data["animal"] = data["food"].map(meat_to_animal)

data

# %% Replacement
data = pd.Series([1., -999., 2., -999., -1000., 3.])
data, data.replace([-999, -1000], np.nan), data.replace({-999:0, -1000:1})

# %% Renaming Axis with Map
data = pd.DataFrame(np.arange(12).reshape((3, 4)), index=["Ohio", "Colorado", "New York"], columns=["one", "two", "three", "four"])

def transform(word):
    return word[:4].upper()

data.index = data.index.map(transform)
data = data.rename(columns=str.upper)
data

# %% Binning
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
names = ["Youth", "Young Adult", "Adult", "Senior"]

age_groups = pd.cut(ages, bins, labels=names)

age_groups.categories, age_groups.codes