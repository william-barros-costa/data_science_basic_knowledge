# %% import 
import pandas as pd
from pandas import Series, DataFrame

# %% Creating a series
serie = Series([1,2,4])

serie, serie.array, serie.index
# %% Specify Index
serie = Series([1,2,4], index=["a", "b", "c"])

serie, serie.index, serie["a"]
# %% Slicing and using numpy
import numpy as np
serie = Series([1,2,4], index=["a", "b", "c"])

serie[serie > 1], serie * 2, np.exp(serie)

# %% In operator 
serie = Series([1,2,4], index=["a", "b", "c"])

"b" in serie, "d" in serie

# %% Dictionaries
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
serie = Series(sdata)

serie, serie.to_dict()

# %% NaN
indexes = ["Ohio", "Texas", "Oregon", "California"]
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
serie_index = Series(sdata, index=indexes)
serie = Series(sdata)

serie, serie_index, serie_index.isna(), serie_index.notna(), serie + serie_index

# %% Naming
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
serie = Series(sdata)

serie.name = "Population"
serie.index.name= "State"

serie

# %% Dataframe
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
        "year": [2000, 2001, 2002, 2001, 2002, 2003],
        "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = DataFrame(data)
frame, frame.head(), frame.tail()

# %% columns
column_names = ["state", "year", "pop", "Debt"] # Needs tho have same name as ditionary
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
        "year": [2000, 2001, 2002, 2001, 2002, 2003],
        "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = DataFrame(data, columns=column_names)
frame.head()

# %% Retrieving rows
frame.iloc[1], frame.loc[1]

# %% Retrieving series
frame["pop"]

# %% Replacing Columns
frame["Debt"] = 12
frame["Debt"] = np.arange(6)
frame["Debt"] = Series([1,2, 4], index=["two", "four", "five"])
frame["Debt"] = Series([1,2,1,2,1,2])
frame["Debt"] = Series([11, 22, 44], index=[1, 4, 5])
frame["Eastern"] = frame["state"] == "Ohio"

frame
# %% Removing Columns
del frame["Eastern"]

frame.head()

# %% Nested Dictionaries
populations = {"Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6}, "Nevada": {2001: 2.4, 2002: 2.9}}

frame = DataFrame(populations)
frame, frame.T

# %% Naming
frame.columns.name = "State"
frame.index.name = "Year"

frame
# %% To numpy
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
        "year": [2000, 2001, 2002, 2001, 2002, 2003],
        "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = DataFrame(data)

array = frame.to_numpy()
array

# %% Creating Indexes
index = pd.Index(np.arange(3, -1, -1))
index

serie = Series([1,2,4,5], index=index)
serie

# %% Indexes can have duplicates
index = pd.Index(["a", "b", "a", "c"])
serie = Series([1,2,4,5], index=index)
serie["a"], "a" in serie

# %% Index Operations
index = pd.Index(["a", "b", "a", "c"])

index, \
index.append(pd.Index(["d"])), \
index.difference(pd.Index(["a"])), \
index.intersection(pd.Index(["a", "b"])), \
index.union(pd.Index(["d"])), \
index.isin(pd.Index(["a"])), \
index.delete(1), \
index.drop("c"), \
index.insert(1, "f"), \
index.is_monotonic_decreasing, \
index.is_monotonic_increasing, \
index.is_unique, \
index.unique()

# %% Reindexing
serie = Series([4.5, 7.2, -5.3, 3.6], index=["d", "b", "a", "c"])

serie_reindex = serie.reindex(["a", "b", "c", "d", "e"])
serie, serie_reindex

# %% Forward Filling
serie = Series([-4, 5, -8], index=[0,2,4])

serie_reindex = serie.reindex(np.arange(6), method="ffill")
serie, serie_reindex

# %% Reindexing Columns
frame = DataFrame(np.arange(9).reshape((3, 3)), index=["a", "c", "d"], columns=["Ohio", "Texas", "California"])
reindex_rows = frame.reindex(index = ["a", "b", "c", "d"])
reindex_columns = frame.reindex(columns = ["Texas", "Utah", "California"])


frame, \
reindex_rows,\
reindex_columns

# %% drop Rows
frame = DataFrame(np.arange(9).reshape((3, 3)), index=["a", "c", "d"], columns=["Ohio", "Texas", "California"])
dropped_row = frame.drop("a")
dropped_row = frame.drop(index=["a"])
dropped_column = frame.drop(columns=["Ohio"])
dropped_column = frame.drop("Ohio", axis=1)
dropped_column = frame.drop(["Ohio", "California"], axis="columns")

frame, dropped_row, dropped_column

# %% Indexing with loc
frame = DataFrame([1,2,4], index=[0,2,1])
frame.loc[[0,1]] # Uses Indexes to Index Dataframe

# %% Indexing with iloc
frame = DataFrame([1,2,4], index=[0,2,1])
frame.iloc[[0,1]] # Uses Position to Index Dataframe

# %% Indexing All together
frame = DataFrame(np.arange(9).reshape((3, 3)), index=["a", "c", "d"], columns=["Ohio", "Texas", "California"])

# Slices Columns
frame["Ohio"], \
# Slices Rows by index
frame.loc["d"], \
# Slices Rows by location
frame.iloc[2], \
# Indexes row by Index and Column
frame.loc["d", ["Ohio", "California"]], \
# Index by row by location and column
frame.iloc[2, [0,2]], \
# Retrieve Value by Name
frame.at["d", "Ohio"],\
# Retrieve Value by Position
frame.iat[2, 0]

# %% Fill Values
df1 = pd.DataFrame(np.arange(12.).reshape((3, 4)),
                      columns=list("abcd"))
df2 = pd.DataFrame(np.arange(20.).reshape((4, 5)),
                       columns=list("abcde"))
df1,df2,df1 + df2, df1.add(df2, fill_value=0)

# %% Reindex fill
df1.reindex(columns=df2.columns, fill_value=0)

# %% Operations between Dataframe and Series
frame = DataFrame(np.arange(12.).reshape((3,4)))
serie = frame.iloc[0]
frame, serie, frame - serie

# %% Subtraction
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list("bde"), index=["Utah", "Ohio", "Texas", "Oregon"])
serie = Series(np.arange(3), index=["b", "e", "f"])

frame, serie, frame - serie

# %% Subtraction over index
frame = pd.DataFrame(np.arange(12.).reshape((4, 3)), columns=list("bde"), index=["Utah", "Ohio", "Texas", "Oregon"])
serie = frame["d"]

frame, serie, frame.sub(serie, axis="index")

# %% Mapping functions
def f1(x: np.array):
        return x.max() - x.min()

frame = pd.DataFrame(np.random.standard_normal((4, 3)), columns=list("bde"), index=["Utah", "Ohio", "Texas", "Oregon"])

np.abs(frame), frame.apply(f1), frame.apply(f1, axis=1)

# %% Mapping Functions with Series as output
def f2(x):
        return Series([x.min(), x.max()], index=["min", "max"])

frame.apply(f2)

# %% Elemente Wise
def f3(x):
        return f"{x:.2f}"

frame, frame.applymap(f3)

# %% Sorting
frame = Series(np.arange(4), index=["d", "a", "b", "c"])
frame, frame.sort_index()

# %% Sorting columns
frame = pd.DataFrame(np.arange(8).reshape((2, 4)), index=["three", "one"], columns=["d", "a", "b", "c"])

frame, frame.sort_index(axis="columns", ascending=False)

# %% Sorting a series
serie = Series([4, np.nan, 2,5,1])

serie, serie.sort_values(na_position="first")

# %% Ranking (Order. Ties are handled according to method)
serie = Series([7, -5, 4, 2, 6, 0, 7])

serie.rank(), serie.rank(ascending=False), \
serie.rank(method="dense"),
serie.rank(method="max"), \
serie.rank(method="min"), \
serie.rank(method="first")