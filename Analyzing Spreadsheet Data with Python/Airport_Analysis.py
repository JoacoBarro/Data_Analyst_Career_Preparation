import openpyxl
import pandas as pd

df_original = pd.read_excel('Airport_Data.xlsx')
df_facilities = pd.read_excel('Airport_Data.xlsx', sheet_name='Facilities')

# Displays the contents of df_facilities
print(df_facilities)

# Returns unique values in the Type column of df_facilities
print(df_facilities["Type"].unique())

# Returns boolean. If Type is SEAPLANE BASE, returns True
print(df_facilities["Type"] == "SEAPLANE BASE")

# Filters and displays rows where Type is SEAPLANE BASE
print(df_facilities[df_facilities["Type"] == "SEAPLANE BASE"])

# Sorts df_facilities by ARPElevation in descending order and displays the result with top 5 entries
print(df_facilities.sort_values(by = "ARPElevation", ascending = False).head(n = 5))

# Calculates and displays mean, max, and min of ARPElevation column
print(df_facilities["ARPElevation"].mean())
print(df_facilities["ARPElevation"].max())
print(df_facilities["ARPElevation"].min())

# Filters df_facilities for SEAPLANE BASE and exports to seaplane.xlsx
df_seaplane = df_facilities[df_facilities["Type"] == "SEAPLANE BASE"]
df_seaplane.to_excel("./Seaplane.xlsx")


