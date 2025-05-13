import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

data=fetch_california_housing()

df=pd.DataFrame(data.data,columns=data.feature_names)
df['Target']=data.target

variable_meaning={
 "MedInc":"Median income in block group",
 "HouseAge":"Median house age in block group",
 "AveRooms":"Average number of rooms per household",
 "AveBedrms":"Average number of bedrooms per household",
 "Population":"Population of block group",
 "AveOccup":"Average number of household members",
 "Latitude":"Latitude of block group",
 "Longitude":"Longitude of block group",
 "Target":"Median house value(in $100,000s)"
}

variable_df=pd.DataFrame(list(variable_meaning.items()),
columns=["Feature","Description"])
print("\nVariable Meaning Table:")
print(variable_df)

print("\nBasic Information about Dataset:")
print(df.info())
print("\nFirst Five Rows of Dataset:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\n Summary Explaination")
print(summary_explaination)

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

plt.figure(figsize=(12,8))
df.hist(figsize=(12,8),bins=30,edgecolor='black')
plt.suptitle("Feature Distributions",fontsize=16)
plt.show()

plt.figure(figsize=(12,8))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplots of Features to Identify Outliers")
plt.show()

plt.figure(figsize=(10,6))
corr_matrix=df.corr()
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

sns.pairplot(df[['MedInc','HouseAge','AveRooms','Target']],
diag_kind='kde')
plt.show()

print("\nKey Insights:") 
print("1. The dataset has", df.shape[0], "rows and", df.shape[1], "columns.") 
print("2. No missing values were found in the dataset.") 
print("3. Histograms show skewed distributions in some features like 
'MedInc'.") 
print("4. Boxplots indicate potential outliers in 'AveRooms' and 
'AveOccup'.") 
print("5. Correlation heatmap shows 'MedInc' has the highest correlation with 
house prices.")
