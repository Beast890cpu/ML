import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.preprocessing import StandardScaler 
 
import warnings 
warnings.filterwarnings('ignore') 
  
data = pd.read_csv(r"C:\Users\Admin\OneDrive\Documents\Machine Learning Lab\Datasets\Boston housing dataset.csv") 

data.head()

data.shape 

  data.info()

  data.nunique() 

  data.CHAS.unique() 

  data.ZN.unique() 

  data.isnull().sum() 

  data.duplicated().sum() 

  df = data.copy() 

  df.isnull().sum()

  df.head() 

  df.describe().T 

  for i in df.columns: 
    plt.figure(figsize=(6,3)) 
 
    plt.subplot(1, 2, 1) 
    df[i].hist(bins=20, alpha=0.5, color='b',edgecolor='black') 
    plt.title(f'Histogram of {i}') 
    plt.xlabel(i) 
    plt.ylabel('Frequency') 
 
    plt.subplot(1, 2, 2) 
    plt.boxplot(df[i], vert=False) 
    plt.title(f'Boxplot of {i}') 
 
    plt.show() 

  corr = df.corr(method='pearson') 
plt.figure(figsize=(10, 8)) 
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5) 
plt.xticks(rotation=90, ha='right') 
plt.yticks(rotation=0) 
plt.title("Correlation Matrix Heatmap") 
plt.show() 
 
scale = StandardScaler() 
X_scaled  = scale.fit_transform(X)  
X_train, X_test, y_train, y_test = train_test_split(X_scaled , y, 
test_size=0.2, random_state=42)  
model = LinearRegression() 
  
model.fit(X_train, y_train) 
LinearRegression()  
y_pred = model.predict(X_test) 
y_pred

mse = mean_squared_error(y_test, y_pred) 
  
rmse = np.sqrt(mse)

  r2 = r2_score(y_test, y_pred) 
print(f'Mean Squared Error: {mse}') 
print(f'Root Mean Squared Error: {rmse}') 
print(f'R-squared: {r2}') 
