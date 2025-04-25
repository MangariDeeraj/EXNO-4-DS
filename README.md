# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
# FUTURE SCALING:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/afd0adc3-608c-4479-9563-b01c8ebb1973)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/1824728a-de7e-4736-8b27-8d0ad0ecc63e)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/1bccdb22-ad70-4538-a357-f1c7fb03e019)
```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals

```
![image](https://github.com/user-attachments/assets/3673c6ad-5486-4758-a49c-e2f0b2619002)
## STANDARD SCALING
```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
![image](https://github.com/user-attachments/assets/a0cdff0f-de9f-4749-a355-01516f84f166)
```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/71fe623d-382c-4012-b741-e3c19e050bb9)
# MAXIMUM ABSOLUTE SCALING:
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
```
![image](https://github.com/user-attachments/assets/092ac94e-68e0-460b-99d9-dfc11df736f1)
# ROBUST SCALING:
```
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()
```
![image](https://github.com/user-attachments/assets/0f4d1a08-c613-4ec0-80e5-33f09979591b)
```
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/d666f5d6-472c-471f-81a2-45843623f341)
# FEATURE SELECTION:
```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![image](https://github.com/user-attachments/assets/37baf1f4-8e13-4c93-bd60-dba619e9c13c)
```
df
```
![image](https://github.com/user-attachments/assets/328becf7-3d1e-40f4-820f-592bed2bfa18)
```
df.info()
```
![image](https://github.com/user-attachments/assets/44aa7e3d-6456-4f3f-a40d-8109e17032d8)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/730a0f69-63c8-4f70-a6a1-715c6b8d6f1c)

# RESULT:    # INCLUDE YOUR RESULT HERE
