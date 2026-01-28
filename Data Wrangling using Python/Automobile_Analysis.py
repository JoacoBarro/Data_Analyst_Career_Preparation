import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the automobile dataset
filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

# Define the column headers
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

# Read the dataset into a pandas DataFrame
df = pd.read_csv(filename, names = headers)

# Display the first few rows of the DataFrame
print(df.head())

# replace ? for NaN values
df.replace("?", np.nan, inplace=True)
print(df.head())

# Check for missing values in the dataset -- True indicates missing value
missing_data = df.isnull()
print(missing_data.head(5))

# Count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    

# Calculate the average of "normalized-losses" column, ignoring NaN values
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

# Replace NaN values in "normalized-losses" column with the average value
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

# Calculate the average of "bore" column, ignoring NaN values
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)

# Replace NaN values in "bore" column with the average value
df["bore"].replace(np.nan, avg_bore, inplace=True)

# Calculate the average of "stroke" column, ignoring NaN values
avg_stroke=df['stroke'].astype('float').mean(axis=0)
print("Average of stroke:", avg_stroke)

# Replace NaN values in "stroke" column with the average value
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

# Calculate the average of "horsepower" column, ignoring NaN values
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)

# Replace NaN values in "horsepower" column with the average value
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

# Calculate the average of "peak-rpm" column, ignoring NaN values
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)

# Replace NaN values in "peak-rpm" column with the average value
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

print(" ")
print(df['num-of-doors'].value_counts())
print(df['num-of-doors'].value_counts().idxmax())

# Replace NaN values in "num-of-doors" column with the most frequent value
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# Drop rows where "price" is NaN
df.dropna(subset=["price"], axis=0, inplace=True)
# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

print(" ")
print("----------------------------- DATAFRAME WITH NO MISSING VALUES ----------------------------- ")
print("Dataframe with no missing values:")
print(df.head())
print(df.isnull().sum())
print("----------------------------- DATAFRAME WITH NO MISSING VALUES ----------------------------- ")
print(" ")

print("Data types of each column:")
print(df.dtypes)

# Convert data types to appropriate formats
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

print("Data types after conversion:")
print(df.dtypes)

print(" ")
print("----------------------------- DATA STANDARIZATION ----------------------------- ")
print("----------------------------- DATA STANDARIZATION ----------------------------- ")
print("----------------------------- DATA STANDARIZATION ----------------------------- ")
print("----------------------------- DATA STANDARIZATION ----------------------------- ")
print(" ")

print(df.head())

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

print(df.head())

# Transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# Rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)

print(df.head())

print(" ")
print("----------------------------- DATA NORMALIZATION ----------------------------- ")
print("----------------------------- DATA NORMALIZATION ----------------------------- ")
print("----------------------------- DATA NORMALIZATION ----------------------------- ")
print("----------------------------- DATA NORMALIZATION ----------------------------- ")
print(" ")

# Normalize "length" and "width" columns
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

# Show the normalized columns (range between 0 and 1)
print(df[["length","width","height"]].head())

print(" ")
print("----------------------------- BINNING ----------------------------- ")
print("----------------------------- BINNING ----------------------------- ")
print("----------------------------- BINNING ----------------------------- ")
print("----------------------------- BINNING ----------------------------- ")
print(" ")

df["horsepower"]=df["horsepower"].astype(int, copy=True)

# Plotting the histogram of horsepower
plt.hist(df["horsepower"])
plt.xlabel("Horsepower")
plt.ylabel("Count")
plt.title("Horsepower Bins - 1")
plt.show()

# Create bins for horsepower
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)

# Create group names for the bins
group_names = ['Low', 'Medium', 'High']

# Bin the "horsepower" column into the new "horsepower-binned" column
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)

# Check the number of vehicles in each bin
df["horsepower-binned"].value_counts()

# Plot the distribution of each bin
plt.bar(group_names, df["horsepower-binned"].value_counts())
plt.xlabel("Horsepower")
plt.ylabel("Count")
plt.title("Horsepower Bins - 2")
plt.show()

# Draw historgram of attribute "horsepower" with bins = 3
plt.hist(df["horsepower"], bins = 3)
plt.xlabel("Horsepower")
plt.ylabel("Count")
plt.title("Horsepower Bins - 3")
plt.show()

print(" ")
print("----------------------------- INDICATOR VARIABLE (Dummy Variable) ----------------------------- ")
print("----------------------------- INDICATOR VARIABLE (Dummy Variable) ----------------------------- ")
print("----------------------------- INDICATOR VARIABLE (Dummy Variable) ----------------------------- ")
print("----------------------------- INDICATOR VARIABLE (Dummy Variable) ----------------------------- ")
print(" ")

# Get indicator variables for "fuel-type" column
print(df.columns)
print(" ")

# Create dummy variables for "fuel-type" column
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print(dummy_variable_1.head())
print(" ")

# Rename the columns for clarity
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
print(dummy_variable_1.head())
print(" ")

# Merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# Drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

# Display the first few rows of the updated DataFrame
print(df.head())

# Get indicator variables of aspiration and assign it to data frame "dummy_variable_2"
dummy_variable_2 = pd.get_dummies(df['aspiration'])

# Change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# Show first 5 instances of data frame "dummy_variable_2"
dummy_variable_2.head()

# Merge the new dataframe to the original datafram
df = pd.concat([df, dummy_variable_2], axis=1)

# Drop original column "aspiration" from "df"
df.drop('aspiration', axis = 1, inplace=True)

print(" ")
print(" ----------------------------- FINAL DATAFRAME ----------------------------- ")
print(" ")
print(df.head())
print(" ")
print(" ----------------------------- FINAL DATAFRAME ----------------------------- ")
print(" ")

# Save the cleaned DataFrame to a CSV file
df.to_csv('clean_df.csv')

