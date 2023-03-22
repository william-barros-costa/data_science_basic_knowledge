# %% Imports
from pandas import DataFrame, Series
import pandas as pd
import numpy as np

# Hierarchical Indexing

# %% First Example
serie = Series(np.random.uniform(size=9),
               index=[["a", "a", "a", "b", "b", "c", "c", "d", "d"],
                      [1, 2, 3, 1, 3, 1, 2, 2, 3]]
               )

serie, serie.index, serie["b":"d"], serie.loc[[
    "b", "d"]], serie.loc[:, 2], serie.unstack()

# %% Second Example
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
                     columns=[["Ohio", "Ohio", "Colorado"],
                              ["Green", "Red", "Green"]])
frame.index.names = ["key 1", "key 2"]
frame.columns.names = ["state", "color"]

frame, frame.index.nlevels, frame.columns.nlevels

# %% Create MultiIndex
multi_index = pd.MultiIndex.from_arrays([
    ["Ohio", "Ohio", "Colorado"], ["Green", "Red", "Green"]
], names=["state", "color"])

multi_index, multi_index.swaplevel("state", "color")

# %% Sorting Indexes
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
                     columns=[["Ohio", "Ohio", "Colorado"],
                              ["Green", "Red", "Green"]])
frame.index.names = ["key 1", "key 2"]
frame.columns.names = ["state", "color"]

frame, frame.sort_index(level=1), frame.swaplevel(
    0, 1), frame.swaplevel(0, 1).sort_index(level=0)

# %% Summary statistics by Level
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
                     columns=[["Ohio", "Ohio", "Colorado"],
                              ["Green", "Red", "Green"]])
frame.index.names = ["key 1", "key 2"]
frame.columns.names = ["state", "color"]

frame, frame.groupby(level="key 2").sum(), frame.groupby(
    level="color", axis="columns").sum()

# %% Indexing with Column
frame = pd.DataFrame({"a": range(7), "b": range(7, 0, -1),
                      "c": ["one", "one", "one", "two", "two",
                            "two", "two"],
                      "d": [0, 1, 2, 0, 1, 2, 3]})

frame, frame.set_index(["c", "d"]), frame.set_index(
    ["c", "d"], drop=False), frame.set_index(["c", "d"]).reset_index()

# Combining and merging Datasets

# %% Merge (inner)

frame_1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "a", "b"],
                       "data1": pd.Series(range(7), dtype="Int64")})
frame_2 = pd.DataFrame({"key": ["a", "b", "d"],
                        "data2": pd.Series(range(3), dtype="Int64")})

frame_1, frame_2, frame_1.merge(frame_2), pd.merge(
    frame_1, frame_2, left_on="data1", right_on="data2")

# %% Merge (outer, left, right)
frame_1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "a", "b"],
                       "data1": pd.Series(range(7), dtype="Int64")})
frame_2 = pd.DataFrame({"key": ["a", "b", "d"],
                        "data2": pd.Series(range(3), dtype="Int64")})

pd.merge(frame_1, frame_2, how="outer"), pd.merge(
    frame_1, frame_2, how="left"), pd.merge(frame_1, frame_2, how="right")

# %% Merge (on)
left = pd.DataFrame({"key1": ["foo", "foo", "bar"],
                     "key2": ["one", "two", "one"],
                     "lval": pd.Series([1, 2, 3], dtype='Int64')})

right = pd.DataFrame({"key1": ["foo", "foo", "bar", "bar"],
                      "key2": ["one", "one", "one", "two"],
                      "rval": pd.Series([4, 5, 6, 7], dtype='Int64')})

left, right, pd.merge(left, right, on=["key1", "key2"], how="outer"), pd.merge(
    left, right, on="key1", how="outer", suffixes=('_left', '_right'))

# %% Merging on Index
left1 = pd.DataFrame({"key": ["a", "b", "a", "a", "b", "c"],
                      "value": pd.Series(range(6), dtype="Int64")})

right1 = pd.DataFrame({"group_val": [3.5, 7]}, index=["a", "b"])

pd.merge(left1, right1, left_on="key", right_index=True), \
    pd.merge(left1, right1, left_on="key", right_index=True, how="outer")

# %% Merge on 1 MultiIndex
lefth = pd.DataFrame({"key1": ["Ohio", "Ohio", "Ohio",
                               "Nevada", "Nevada"],
                      "key2": [2000, 2001, 2002, 2001, 2002],
                      "data": pd.Series(range(5), dtype="Int64")})

righth_index = pd.MultiIndex.from_arrays([
    ["Nevada", "Nevada", "Ohio", "Ohio", "Ohio", "Ohio"],
    [2001, 2000, 2000, 2000, 2001, 2002]
])

righth = pd.DataFrame({
    "event1": pd.Series([0, 2, 4, 6, 8, 10], dtype="Int64", index=righth_index),
    "event2": pd.Series([1, 3, 5, 7, 9, 11], dtype="Int64", index=righth_index)
})

pd.merge(lefth, righth, left_on=["key1", "key2"], right_index=True), \
    pd.merge(lefth, righth, left_on=[
             "key1", "key2"], right_index=True, how="outer"),

# %% Merge on both MultiIndex
left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
                     index=["a", "c", "e"],
                     columns=["Ohio", "Nevada"]).astype("Int64")

right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
                      index=["b", "c", "d", "e"],
                      columns=["Missouri", "Alabama"]).astype("Int64")

pd.merge(left2, right2, left_index=True, right_index=True, how="outer")

# %% Simpler form to merge by index
left2.join(right2, how="outer")

# %% Simple Concatenation

array = np.arange(12).reshape((3, 4))
np.concatenate([array, array], axis=1)

# %% Several Series Concatenation

s1 = pd.Series([0, 1], index=["a", "b"], dtype="Int64")
s2 = pd.Series([2, 3, 4], index=["c", "d", "e"], dtype="Int64")
s3 = pd.Series([5, 6], index=["f", "g"], dtype="Int64")

pd.concat([s1, s2, s3]), pd.concat([s1, s2, s3], axis="columns")

# %% Concatenation with Inner join

s4 = pd.concat([s1, s3])
s1, s4, pd.concat([s1, s4], join="inner", axis="columns")

# %% Keys Argument
df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=["a", "b", "c"],
                   columns=["one", "two"])

df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=["a", "c"],
                   columns=["three", "four"])

df1, df2, pd.concat([df1, df2], keys=["level1", "level2"]), \
    pd.concat([df1, df2], keys=["level1", "level2"], axis="columns"), \
    pd.concat({"level1": df1, "level2": df2}, axis="columns")

# %% Ignore Index
df1 = pd.DataFrame(np.random.standard_normal((3, 4)),
                   columns=["a", "b", "c", "d"])
df2 = pd.DataFrame(np.random.standard_normal((2, 3)),
                   columns=["b", "d", "a"])

df1, df2, pd.concat([df1, df2]), pd.concat([df1, df2], ignore_index=True)

# %% Combining Data with Overlap
a = pd.Series([np.nan, 2.5, 0.0, 3.5, 4.5, np.nan],
              index=["f", "e", "d", "c", "b", "a"])
b = pd.Series([0., np.nan, 2., np.nan, np.nan, 5.],
              index=["a", "b", "c", "d", "e", "f"])

# Np.Where does not care about Indexes while combine_first does.
np.where(pd.isna(a), b, a), a.combine_first(b)

# %% Reshaping with Hierarchical Indexing
data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(["Ohio", "Colorado"], name="state"),
                    columns=pd.Index(["one", "two", "three"],
                                     name="number"))
result = data.stack()

data, result, result.unstack(),
result.unstack(level=0), result.unstack(level="state")

# %% Unstack introduces missing data
s1 = pd.Series([0, 1, 2, 3], index=["a", "b", "c", "d"], dtype="Int64")
s2 = pd.Series([4, 5, 6], index=["c", "d", "e"], dtype="Int64")
data2 = pd.concat([s1, s2], keys=["one", "two"])

data2.unstack(), data2.unstack().stack(dropna=False)

# %% Stacking with Index name
df = pd.DataFrame({"left": result, "right": result + 5},
                  columns=pd.Index(["left", "right"], name="side"))

df, df.unstack(level="state"), df.unstack(level="state").stack("side")

# %% Pivoting Long to Wide
location = r"C:\Users\William\Desktop\repository\data_science\resources\macrodata.csv"
data = pd.read_csv(location)

data = data.loc[:, ["year", "quarter", "realgdp", "infl", "unemp"]]

periods = pd.PeriodIndex(year=data.pop("year"),
                         quarter=data.pop("quarter"),
                         name="date")

data.index = periods.to_timestamp("D")

data.reindex(columns=["realgdp", "infl", "unemp"])
data.columns.name = "item"

long_data = data.stack().reset_index().rename(columns={0: "value"})

# Pivoted Table, initial Table
long_data, long_data.pivot(index="date", columns="item", values="value").head()

# %% Pivoting Wide to Long

df = pd.DataFrame({"key": ["foo", "bar", "baz"],
                   "A": [1, 2, 3],
                   "B": [4, 5, 6],
                   "C": [7, 8, 9]})
melted = pd.melt(df, id_vars="key")

melted, melted.pivot(index="key", values="value", columns="variable")
