# %% Imports
import pandas as pd
import numpy as np
import re
from pandas import Series, DataFrame

# %% Types of missing data
float_serie = Series([1.2, -3.5, np.nan, None, 0])

float_serie, float_serie.isna()

# %% Filter missing Data
data = pd.Series([1, np.nan, 3.5, np.nan, 7])

data.dropna(), data[data.notna()]

# %% Remove row with all missing data
data = pd.DataFrame([[1., 6.5, 3.], [1., np.nan, np.nan], [
                    np.nan, np.nan, np.nan], [np.nan, 6.5, 3.]])

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
data = pd.DataFrame(
    {"k1": ["one", "two"] * 3 + ["two"], "k2": [1, 1, 2, 3, 3, 4, 4]})
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
data, data.replace([-999, -1000], np.nan), data.replace({-999: 0, -1000: 1})

# %% Renaming Axis with Map
data = pd.DataFrame(np.arange(12).reshape((3, 4)), index=[
                    "Ohio", "Colorado", "New York"], columns=["one", "two", "three", "four"])


def transform(word):
    return word[:4].upper()


data.index = data.index.map(transform)
data = data.rename(columns=str.upper)
data

# %% Binning with categories
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
names = ["Youth", "Young Adult", "Adult", "Senior"]

age_groups = pd.cut(ages, bins, labels=names)

age_groups.categories, age_groups.codes

# %% Binning with number of bins
data = np.random.uniform(size=20)

bins = pd.cut(data, 4, precision=2)  # Precision -> decimal cases

bins

# %% Binning with quantiles
data = np.random.standard_normal(1000)

quartiles = pd.qcut(data, 4, precision=2)

quartiles, pd.value_counts(quartiles)

# %% Binning with custom quantiles
data = np.random.standard_normal(1000)

quartiles = pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.], precision=2)

quartiles, pd.value_counts(quartiles)

# %% Information about data
data = DataFrame(np.random.standard_normal((1000, 4)))
data.describe(), data.info()

# %% Filter data in Indexes
data = DataFrame(np.random.standard_normal((1000, 4)))

col = data[2]

col[col.abs() > 3]

# %% Filter data in dataFrame
data = DataFrame(np.random.standard_normal((1000, 4)))

data[(data.abs() > 3).any(axis="columns")]

# %% Change values based on filter
data = DataFrame(np.random.standard_normal((1000, 4)))

data[data.abs() > 3] = np.sign(data) * 3

data.describe()

# %% Permutations for rows (Creates a new order for data)
frame = DataFrame(np.arange(5 * 7).reshape((5, 7)))

permutation = np.random.permutation(5)

permutation, frame.iloc[permutation], frame.take(permutation), frame

# %% Permutation for columns
frame = DataFrame(np.arange(5 * 7).reshape((5, 7)))
permutation = np.random.permutation(7)

permutation, frame.iloc[:, permutation], frame.take(
    permutation, axis="columns"), frame

# %% Get Random sample
frame = DataFrame(np.arange(5 * 7).reshape((5, 7)))

frame.sample(n=3), frame.sample(n=3, axis="columns")

# %% Get Sample inplace
choices = Series([5, 7, -1, 6, 4])

choices.sample(n=10, replace=True)

# %% Create Dummy Variable
frame = DataFrame(["b", "b", "a", "c", "a", "b"])

dummies = pd.get_dummies(frame, prefix="key")

frame_with_dummies = frame.join(dummies)

frame_with_dummies

# %% Create dummies for several classes
location = r"C:\Users\William Costa\Documents\repositories\data_science_basic_knowledge\resources\movies.dat"

movies = pd.read_table(location, sep="::", header=None, engine="python",
                       names=["movie_id", "title", "genres"],
                       )

dummies = movies["genres"].str.get_dummies()

movies_with_dummies = movies.join(dummies.add_prefix("Genre_"))

movies_with_dummies.iloc[0]

# %% Dummies with Cut
np.random.seed(12345)

values = np.random.uniform(size=10)
bins = [0., 0.2, 0.4, 0.6, 0.8, 1.]
pd.get_dummies(pd.cut(values, bins))

# String Manipulation

# %% Contains
data = Series({"Dave": "dave@google.com", "Steve": "steve@gmail.com",
               "Rob": "rob@gmail.com", "Wes": np.nan})

data.str.contains("gmail")

# %% Find patterns
pattern = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"

matches = data.str.findall(pattern, flags=re.IGNORECASE)
matches

# %% Some basic functions
matches.str[0], matches.str[0].get(1), matches.str[0].str.get(1)

# %% slicing
data.str[:5]

# %% Extract regex as DataFrame
data.str.extract(pattern, flags=re.IGNORECASE)

# %% STR Functions
data.str.cat(sep="::"),
data.str.contains("@"),
data.str.count("g"),
data.str.endswith(".com"),
data.str.startswith("a"),
data.str.findall("@"),
data.str.get(3),
data.str.isalnum(),
data.str.isalpha(),
data.str.isdecimal(),
data.str.isdigit(),
data.str.islower(),
data.str.isnumeric(),
data.str.isupper(),
data.str.join("::"),
data.str.len(),
data.str.lower(),
data.str.upper(),
data.str.match("s"),
data.str.pad(3, side="left"),
data.str.center(5),
data.str.repeat(2),
data.str.replace("@", "--"),
data.str.slice(3),
data.str.split("@"),
data.str.strip(),
data.str.rstrip(),
data.str.lstrip()

# Categorical Data

# %% Converting Series dummies to categories
values = Series([0, 1, 0, 0] * 2)
dimensions = Series(["apples", "oranges"])

dimensions.take(values)

# %% Converting DataFrame column to type category
fruits = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits)
rng = np.random.default_rng(seed=12345)
frame = pd.DataFrame({'fruit': fruits,
                   'basket_id': np.arange(N),
                   'count': rng.integers(3, 15, size=N),
                   'weight': rng.uniform(0, 4, size=N)},
                  columns=['basket_id', 'fruit', 'count', 'weight'])

categories = frame["fruit"].astype('category')

categories.array.categories, categories.array.codes

# %% Create a simple map
dict(enumerate(categories.array.categories))

# %% Convert DataFrame column to type category in DataFrame
frame['fruit'] = frame['fruit'].astype("category")
frame['fruit']

# %% Array to Category
categories = pd.Categorical(["foo", 'bar', 'baz', 'foo', 'bar'])
categories

# %% Dummies to categories
categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
categories_restored = pd.Categorical.from_codes(codes, categories)

categories_restored

# %% Ordered
categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
categories_restored = pd.Categorical.from_codes(codes, categories, ordered=True)

categories_restored

# Computations with Categoricals

# %% Add categories to cut
rng = np.random.default_rng(seed=12345)
draws = rng.standard_normal(1000)

bins = pd.qcut(draws, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
bins[:10], bins.codes[:10]

# %% Summary Statistics
bins = Series(bins, name='Quartile')

results = (Series(draws).groupby(bins).agg(['count', 'min', 'max']).reset_index())

results

# Categorical Methods

# %% Cat
serie = Series(['a', 'b', 'c', 'd'] * 2)
category_serie = serie.astype('category')
category_serie.cat.codes, category_serie.cat.categories

# %% Set Categories
category_serie_2 = category_serie.cat.set_categories(['a', 'b', 'c', 'd', 'e'])

category_serie.value_counts(), category_serie_2.value_counts()

# %% Remove unused categories
category_serie_3 = category_serie[category_serie.isin(['a', 'b'])]

category_serie_3, category_serie_3.cat.remove_unused_categories()
