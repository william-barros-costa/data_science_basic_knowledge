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

# %% 

