import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)

# Display the first five rows of the dataframe
print(df.head())

# Display the data types of each column 
print(df.dtypes)

# Calculate the correlation between 'bore', 'stroke', 'compression-ratio', and 'horsepower'
# The diagonal elements will be 1 as they represent the correlation of each variable with itself
print(df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr())

print("\n ---------- CONTINUOUS NUMERICAL VARAIBLES ---------- \n")
print("\n ---------- CONTINUOUS NUMERICAL VARAIBLES ---------- \n")
print("\n ---------- CONTINUOUS NUMERICAL VARAIBLES ---------- \n")

# Continuous numerical variables are variables that may contain any value within some range. 
# They can be of type "int64" or "float64". 
# A great way to visualize these variables is by using scatterplots with fitted lines.

# Engine size as potential predictor variable of price
# Positive Linear Relationship
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
plt.title('Engine Size vs Price - Positive Linear Relationship')
# As the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables. 
# Engine size seems like a pretty good predictor of price since the regression line is almost a perfect diagonal line.
plt.show()

# We can examine the correlation between 'engine-size' and 'price' and see that it's approximately 0.87.
print(df[["engine-size", "price"]].corr())


# Highway mpg as potential predictor variable of price
# Negative Linear Relationship
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
plt.title('Highway-mpg vs Price - Negative Linear Relationship')
# As highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship between these two variables.
# Highway mpg could potentially be a predictor of price.
plt.show()

# We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704.
print(df[['highway-mpg', 'price']].corr())

# Peak-rpm as potential predictor variable of price
# Weak Linear Relationship
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
plt.title('Peak-rpm vs Price - Weak Linear Relationship')
# Peak rpm does not seem like a good predictor of the price at all since the regression line is close to horizontal. 
# Also, the data points are very scattered and far from the fitted line, showing lots of variability. 
# Therefore, it's not a reliable variable.
plt.show()

# We can examine the correlation between 'peak-rpm' and 'price' and see it's approximately -0.101616.
print(df[['peak-rpm','price']].corr())


print("\n ---------- CATEGORICAL VARAIBLES ---------- \n")
print("\n ---------- CATEGORICAL VARAIBLES ---------- \n")
print("\n ---------- CATEGORICAL VARAIBLES ---------- \n")

# These are variables that describe a 'characteristic' of a data unit, and are selected from a small group of categories. 
# The categorical variables can have the type "object" or "int64". 
# A good way to visualize categorical variables is by using BOXPLOTS.

sns.boxplot(x="body-style", y="price", data=df)
plt.title('Body Style vs Price')
# Here we can see that the sedan and hatchback body styles have a wider price range,
# while the convertible body style has the least price range.
plt.show()

sns.boxplot(x="engine-location", y="price", data=df)
plt.title('Engine Location vs Price')   
# Here we can see that the engine location at the front has a much wider price range,
# while the engine location at the rear has a lower price range.
plt.show()

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.title('Drive Wheels vs Price')  
# Here we can see that the drive-wheels with 4wd has a much wider price range,
# while the drive-wheels with rwd has a higher price range than fwd.
plt.show()


print("\n ---------- DESCRIPTIVE STATISTICAL ANALYSIS ---------- \n")
print("\n ---------- DESCRIPTIVE STATISTICAL ANALYSIS ---------- \n")
print("\n ---------- DESCRIPTIVE STATISTICAL ANALYSIS ---------- \n")


# The describe function automatically computes basic statistics for all continuous variables. 
# Any NaN values are automatically skipped in these statistics.
# This will show:
    # the count of that variable
    # the mean
    # the standard deviation (std)
    # the minimum valuethe IQR (Interquartile Range: 25%, 50% and 75%)
    # the maximum value

print(df.describe()) # Skips "Object" dtype variables
# To apply the describe function to categorical variables, we need to select only the "object" dtype variables.
print(df.describe(include=['object'])) # Apply describe to categorical variables


print("\n ---------- VALUE CONTS ---------- \n")
print("\n ---------- VALUE CONTS ---------- \n")
print("\n ---------- VALUE CONTS ---------- \n")

# Good way of understanding how many units of each characteristic/variable we have.
print(df['drive-wheels'].value_counts())
# We can convert the series to a dataframe as follows:
print(df['drive-wheels'].value_counts().to_frame())

# We can rename the column name of the dataframe as follows:
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
print(drive_wheels_counts)

# We can also rename the index name of the dataframe as follows:
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

# Let's repeat the above process for the variable 'engine-location'.
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts.head(10))

# After examining the value counts of the engine location, 
# we see that engine location would not be a good predictor variable for the price. 
# This is because we only have three cars with a rear engine and 198 with an engine in the front, 
# so this result is skewed. Thus, we are not able to draw any conclusions about the engine location.

print("\n ---------- BASICS OF GROUPING ---------- \n")
print("\n ---------- BASICS OF GROUPING ---------- \n")
print("\n ---------- BASICS OF GROUPING ---------- \n")

# The "groupby" method groups data by different categories. 
# The data is grouped based on one or several variables, and analysis is performed on the individual groups.

# Let's group the data by 'drive-wheels'.
print(df['drive-wheels'].unique())

# We can select the columns we want to group.
df_group_one = df[['drive-wheels','body-style','price']]
# We can then calculate the average price for each of the different categories of data
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
print(df_group_one)

# From our data, it seems rear-wheel drive vehicles are, on average, the most expensive, 
# while 4-wheel and front-wheel are approximately the same in price.

# You can also group by multiple variables. 
# For example, let's group by both 'drive-wheels' and 'body-style'. 
# This groups the dataframe by the unique combination of 'drive-wheels' and 'body-style'. 
# We can store the results in the variable 'grouped_test1'.

df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(grouped_test1)

# This grouped data is much easier to visualize when it is made into a pivot table. 
# A pivot table is like an Excel spreadsheet, with one variable along the column and another along the row. 
# We can convert the dataframe to a pivot table using the method "pivot" to create a pivot table from the groups.

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
print(grouped_pivot)

# Often, we won't have data for some of the pivot cells. We can fill these missing cells with the value 0, 
# but any other value could potentially be used as well. 
# It should be mentioned that missing data is quite a complex subject and is an entire course on its own

grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
print(grouped_pivot)

# We can also group by 'body-style' alone and find the average price for each body style.
df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
print(grouped_test_bodystyle)

# We can plot the pivot table we created above to visualize the relationship between 'drive-wheels' and 'body-style' with respect to price.
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

# The heatmap plots the target variable (price) proportional to colour with respect to the variables 'drive-wheel' and 'body-style' 
# on the vertical and horizontal axis, respectively. This allows us to visualize how the price is related to 'drive-wheel' and 'body-style'.

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

# From this heatmap, we can confirm our observations from the grouped pivot table.
# We can see that rear-wheel drive vehicles with a convertible body style have the highest average price.


print("\n ---------- CORRELATION AND CAUSATION ---------- \n")
print("\n ---------- CORRELATION AND CAUSATION ---------- \n")
print("\n ---------- CORRELATION AND CAUSATION ---------- \n")

pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  
# Since the p-value is <0.001, the correlation between wheel-base and price
# is statistically significant, although the linear relationship isn't extremely strong (~0.585). 

pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
# Since the p-value is <0.001, the correlation between horsepower and price 
# is statistically significant, and the linear relationship is quite strong (~0.809, close to 1).

pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
# Since the p-value is <0.001, the correlation between length and price
# is statistically significant, and the linear relationship is fairly strong (~0.690).

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 
# Since the p-value is <0.001, the correlation between width and price
# is statistically significant, and the linear relationship is fairly strong (~0.751).

pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
# Since the p-value is <0.001, the correlation between curb-weight and price
# is statistically significant, and the linear relationship is quite strong (~0.834).

pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
# Since the p-value is <0.001, the correlation between engine-size and price
# is statistically significant, and the linear relationship is very strong (~0.872).

pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value ) 
# Since the p-value is <0.001, the correlation between bore and price
# is statistically significant, but the linear relationship is weak (~0.543).

pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  
# Since the p-value is <0.001, the correlation between city-mpg and price
# is statistically significant, but the linear relationship is weak (~ -0.686).

pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value ) 
# Since the p-value is <0.001, the correlation between highway-mpg and price
# is statistically significant, but the linear relationship is weak (~ -0.704).

print("\n ---------- ANOVA ---------- \n")
print("\n ---------- ANOVA ---------- \n")
print("\n ---------- ANOVA ---------- \n")

# The Analysis of Variance (ANOVA) is a statistical method used to test whether there are significant 
# differences between the means of two or more groups. ANOVA returns two parameters:

# F-test score: ANOVA assumes the means of all groups are the same, 
# calculates how much the actual means deviate from the assumption, and reports it as the F-test score. 
# A larger score means there is a larger difference between the means.

# P-value: P-value tells how statistically significant our calculated score value is.

# f our price variable is strongly correlated with the variable we are analyzing, 
# we expect ANOVA to return a sizeable F-test score and a small p-value.

grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
print(grouped_test2.head(2))
print(df_gptest)

grouped_test2.get_group('4wd')['price']

# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
print( "ANOVA results: F=", f_val, ", P =", p_val)   
# ANOVA results: F= 67.95406500780399 , P = 3.3945443577151245e-23

# This is a great result with a large F-test score showing a strong correlation and a P-value of almost 0 
# implying almost certain statistical significance. But does this mean all three tested groups are all this highly correlated?

f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
print( "ANOVA results: F=", f_val, ", P =", p_val )

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
print( "ANOVA results: F=", f_val, ", P =", p_val)   

f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
print("ANOVA results: F=", f_val, ", P =", p_val)   

# We notice that ANOVA for the categories 4wd and fwd yields a high p-value > 0.1, 
# so the calculated F-test score is not very statistically significant. 
# This suggests we can't reject the assumption that the means of these two groups are the same, or, in other words, 
# we can't conclude the difference in correlation to be significant

# We now have a better idea of what our data looks like and which variables are important
#  to take into account when predicting the car price. We have narrowed it down to the following variables:

# Continuous numerical variables:
    # Length
    # Width
    # Curb-weight
    # Engine-size
    # Horsepower
    # City-mpg
    # Highway-mpg 
    # Wheel-base
    # Bore

# Categorical variables:
    # Drive-wheels






