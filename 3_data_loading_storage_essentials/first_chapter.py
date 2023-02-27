# %% example
!type "..\resources\ex1.csv" # Prints content of file

# %% Imports
import pandas as pd
from pandas import Series, DataFrame

# %% Read File
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex1.csv"
frame = pd.read_csv(location)

frame

# %% Read File without header
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex2.csv"
frame = pd.read_csv(location, header=None)
frame2 = pd.read_csv(location, names=["a", "b", "c", "d", "message"])

frame, frame2

# %% Read File and Index columns
names = ["a", "b", "c", "d", "message"]
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex2.csv"
frame = pd.read_csv(location, names=names, index_col=["message", "a"])

frame

# %% Read File with variable space separator
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex3.txt"
frame = pd.read_csv(location, sep="\s+")

frame

# %% Read File with Null values
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex5.csv"
frame = pd.read_csv(location, na_values=["NULL"])

frame, frame.isna()

# %% Select NaN Values
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex5.csv"
frame = pd.read_csv(location, keep_default_na=False, na_values=["NA"])

frame, frame.isna()

# %% Show only a few lines
pd.options.display.max_rows = 10
  
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex6.csv"
frame = pd.read_csv(location)

frame

# %% Read only a number of lines 
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex6.csv"
frame = pd.read_csv(location, nrows=5)

frame

# %% Prepare iterator for large files
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex6.csv"
iterator = pd.read_csv(location, chunksize=1000)

type(iterator)

# %% Basic iterator usage
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex6.csv"
iterator = pd.read_csv(location, chunksize=1000)

serie = Series([], dtype="int64")
for i in iterator:
    serie = serie.add(i["key"].value_counts(), fill_value=0)
    
serie.sort_values(ascending=False)

# %% Output Dataframe
input_location = r"C:\Users\William\Desktop\repository\data_science\resources\ex5.csv"
output_location = r"C:\Users\William\Desktop\repository\data_science\data_loading_storage_essentials\simple.csv"

input_location = r"C:\Users\William\Desktop\repository\data_science\resources\ex5.csv"
frame = pd.read_csv(input_location)

frame.to_csv(output_location)

# %% Print to console
import sys
input_location = r"C:\Users\William\Desktop\repository\data_science\resources\ex5.csv"
frame = pd.read_csv(input_location)

frame.to_csv(sys.stdout, sep="|", index=False, header=False),
frame.to_csv(sys.stdout, sep="|", index=False, columns=["a", "b", "c"])

# %% Working with delimiters
import csv
input_location = r"C:\Users\William\Desktop\repository\data_science\resources\ex7.csv"

with open(input_location) as f:
    lines = list(csv.reader(f))

header, values = lines[0], lines[1:]

data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict

# %% Create CSV dialect
class my_dialect(csv.Dialect):
    lineterminator = "\n"
    delimiter = ";"
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL

input_location = r"C:\Users\William\Desktop\repository\data_science\resources\ex7.csv"

with open(input_location) as f:
    csv_lines = list(csv.reader(f, dialect=my_dialect))
    print(csv_lines)

# %% Write CSV with Dialect
location = r"C:\Users\William\Desktop\repository\data_science\data_loading_storage_essentials\csv_writer.csv"

with open(location, "w") as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(("one", "two", "three"))
    writer.writerow(("1", "2", "3"))
    writer.writerow(("4", "5", "6"))
    writer.writerow(("7", "8", "9"))

# %% Import Json
import json
json_object = """
{"name": "Wes",
 "cities_lived": ["Akron", "Nashville", "New York", "San Francisco"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 34, "hobbies": ["guitars", "soccer"]},
              {"name": "Katie", "age": 42, "hobbies": ["diving", "art"]}]
}
"""

result = json.loads(json_object)

result

# %% Convert to Json
as_json = json.dumps(result)

as_json

# %% json to DataFrame
json_object = """
{"name": "Wes",
 "cities_lived": ["Akron", "Nashville", "New York", "San Francisco"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 34, "hobbies": ["guitars", "soccer"]},
              {"name": "Katie", "age": 42, "hobbies": ["diving", "art"]}]
}
"""

result = json.loads(json_object)
frame = DataFrame(result['siblings'], columns=['name', 'age'])

frame

# %% Simpler way to get DataFrame from json
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex8.json"

frame = pd.read_json(location)
frame

# %% Dataframe to Json
frame.to_json(sys.stdout)
print()
frame.to_json(sys.stdout, orient="records")

# %% Read all tables from HTML
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex9.html"
frame_list = pd.read_html(location)

failures = frame_list[0]
failures

# %% Parsing XML
location = r"C:\Users\William\Desktop\repository\data_science\resources\ex19.xm"

xml = pd.read_xml(location)
xml

# %% Install Excel Files in pandas
!pip install openpyxl xlrd

#%% Excel files

location = r"C:\Users\William\Desktop\repository\data_science\resources"
xlsx = pd.ExcelFile(f"{location}\\test.xlsx")

xlsx.sheet_names

# %% Write on Excel file
location = r"C:\Users\William\Desktop\repository\data_science\resources"
writer = pd.ExcelWriter(f"{location}\\test.xlsx")

frame = DataFrame([[1,2,3],[1,2,3],[1,2,3]], columns=["col1", "col2", "col3"], index=["row1", "row2", "row3"])


frame.to_excel(writer, "Sheet1")
writer.save()

# %% Read Excel files
location = r"C:\Users\William\Desktop\repository\data_science\resources"
xlsx = pd.ExcelFile(f"{location}\\test.xlsx")

frame = xlsx.parse(sheet_name="Sheet1", index_col=0)
frame

# %% Another option to read
location = r"C:\Users\William\Desktop\repository\data_science\resources"
frame = pd.read_excel(f"{location}\\test.xlsx", sheet_name="Sheet1", index_col=0)
frame

# There are still ways of interacting with extensions hdf, h5py and even SQL files.