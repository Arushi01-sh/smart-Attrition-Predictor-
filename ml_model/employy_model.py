# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:48:39 2026

#as in this colmn we have in target column more no values and less yes values from which our gives wrong prediction if we predict and therefore if values are less then we have the library smot that will understand the pattern of our data and  will generate data of yes values then we will use that values
@author: sharm_clbnasz
"""
##in classsification we use logistic regression it is same as linear but in this values are descrete
import pandas as pd 
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

## load data 
employ_df=pd.read_csv('employdata.csv')

employ_df.shape
employ_df.info()

##to find mean,min,max,std deviation etc
employ_df_des=employ_df.describe() #if std deviation is 0 it is no benifit to use them but if it is 0.334something then we will take it because every value in the column is same 

##checking because standard deviation is 0
employ_df.StockOptionLevel.unique()
employ_df.OverTime.unique()
employ_df.columns
##drop 
employ_df.drop(['EmployeeCount','StandardHours'],axis=1,inplace=True)

employ_df.nunique()

##employnumber is unique id and ob=ver 18 has only one valy that does not effect model
employ_df.drop(['EmployeeNumber','Over18'],axis=1,inplace=True)

# #checking skewnwss and kurtosis
# #plotting distplot and boxplot
# #-0.5to 0.5 is not skew 
# # 0.5 to 1 is moderatee positive ie right skewed 
# # -0.5to -1 is moderate negative ie left skewed
# # greater thanm 1 is highly positive skewed
# # less than -1 highly negative skewed
# #kurtosis are only for normal distribution  is also for outliers agr sbse ptle normal distribution se jyada koi value ho toh woh hmare outliers hote hain    
# #if value (when we find like this print(f'kurtosis of {i}:', employ_df[i].kurtosis()) ) is more than 3 then it is also showing kurtosis ie outliers 
# # if data is less then dont remove outliers just transforms it and if data is more than just remove the outliers



# for i in employ_df.select_dtypes(np.number): #select_dtypes() selects columns based on their data type and np.number select all numeric data types (int, float, etc.)
#     print(f'skewness of {i}:', employ_df[i].skew())
#     print(f'kurtosis of {i}:', employ_df[i].kurtosis())#to find that whether distinct values are closer to mean 
#     plt.figure(figsize=(12,4))#bhr wala table ie jo white colour ka poora table jiske beech do figure aane hain woj table  
#     plt.subplot(121)#to draw two figure # 1 row 2 columns and 1 figure
#     sns.distplot(employ_df[i])
#     plt.subplot(122)#to draw two figure and this one is second
#     sns.boxplot(employ_df[i])
#     plt.show()

# #or another method - from scipy.stats import skew , kurtosis 
# #and then write skew (df[''])
    
# normal_distribution=['age','DailyRate','HourlyRate','MonthlyRate']
# right_skew_column=['NumCompaniesWorked','TotalWorkingYears','YearsAtCompany','DistanceFromHome','MonthlyIncome','PercentSalaryHike']
# skewed_not_aply_box_cox=['DistanceFromHome','MonthlyIncome','PercentSalaryHike']
# catagorical=['Education','EnvironmentSatisfaction','JobInvolvement','JobLevel','JobSatisfaction','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TrainingTimesLastYear','WorkLifeBalance','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']
# employ_df["NumCompaniesWorked"].unique()

# #transforming outliers
# transform_data=pd.DataFrame()
# #here we use only oine formula np log to transform all the columns 
# for i in right_skew_column:
#     transform_data[i]=np.log(employ_df[i])
# # chcking if np.log transform all the columns with outliers correctly
# for i in right_skew_column:
#     print(f'skewness of {i}:', employ_df[i].skew())
#     print(f'kurtosis of {i}:', employ_df[i].kurtosis())
#     print(f'skewness of {i}:', transform_data[i].skew())
#     print(f'kurtosis of {i}:', transform_data[i].kurtosis()) 
    
#     plt.figure(figsize=(15,6)) 
#     plt.subplot(221)
#     sns.distplot(employ_df[i])
#     plt.subplot(222)
#     sns.boxplot(employ_df[i])
#     #plt.show()
    
#     #plt.figure(figsize=(10,6))  
#     plt.subplot(223)
#     sns.distplot(transform_data[i])
#     plt.subplot(224)
#     sns.boxplot(transform_data[i])
#     plt.show()
    
# #as we see that three columns do not transform correctly , it transform values to infinity therefore again chckinmg which method is suitable for which column  
transform_data=pd.DataFrame()#forming new dataframe withy same name 
from sklearn.preprocessing import PowerTransformer


# #log transform all the columns  by using all the methods 
# for i in right_skew_column:
#     transform_data[f'{i}_log']=np.log(employ_df[i])
#     transform_data[f'{i}_log1p']=np.log1p(employ_df[i])
#     transform_data[f'{i}_sqrt']=np.sqrt(employ_df[i])
#     transform_data[f'{i}_cbrt']=np.cbrt(employ_df[i])
#     transform_data[f'{i}_yeo_johnson']=pt1.fit_transform(employ_df[[i]])

# #as box_cox do not apply for all the columns 
# for i in skewed_not_aply_box_cox:
#     transform_data[[f'{i}_box_cox']]=pt.fit_transform(employ_df[[i]])

# #checvking skewmess and kurtoess of all the transforming columns to see which method is mode suitable 

# employ_df['NumCompaniesWorked'].skew()   
# employ_df['NumCompaniesWorked'].kurtosis()

# transform_data['NumCompaniesWorked_log'].skew()
# transform_data['NumCompaniesWorked_log'].kurtosis()
# transform_data['NumCompaniesWorked_log1p'].skew()
# transform_data['NumCompaniesWorked_log1p'].kurtosis()
# transform_data['NumCompaniesWorked_sqrt'].skew()
# transform_data['NumCompaniesWorked_sqrt'].kurtosis()
# transform_data['NumCompaniesWorked_cbrt'].skew()
# transform_data['NumCompaniesWorked_cbrt'].kurtosis()
# # transform_data['NumCompaniesWorked_box_cox'].skew()
# # transform_data['NumCompaniesWorked_box_cox'].kurtosis()
# transform_data['NumCompaniesWorked_yeo_johnson'].skew()
# transform_data['NumCompaniesWorked_yeo_johnson'].kurtosis()
# #checking total working years and like them checking all the columns that have outliers ie are in  right_skew_column
# employ_df['YearsAtCompany'].skew()   
# employ_df['YearsAtCompany'].kurtosis()
  
# transform_data['YearsAtCompany_log'].skew()
# transform_data['YearsAtCompany_log'].kurtosis()
# transform_data['YearsAtCompany_log1p'].skew()
# transform_data['YearsAtCompany_log1p'].kurtosis()
# transform_data['YearsAtCompany_sqrt'].skew()
# transform_data['YearsAtCompany_sqrt'].kurtosis()
# transform_data['YearsAtCompany_cbrt'].skew()
# transform_data['YearsAtCompany_cbrt'].kurtosis()
# #transform_data['YearsAtCompany_box_cox'].skew()
# #transform_data['YearsAtCompany_box_cox'].kurtosis()
# transform_data['YearsAtCompany_yeo_johnson'].skew()
# transform_data['YearsAtCompany_yeo_johnson'].kurtosis()

# employ_df['PercentSalaryHike'].skew()   
# employ_df['PercentSalaryHike'].kurtosis()
# transform_data['PercentSalaryHike_log'].skew()
# transform_data['PercentSalaryHike_log'].kurtosis()
# transform_data['PercentSalaryHike_log1p'].skew()
# transform_data['PercentSalaryHike_log1p'].kurtosis()
# transform_data['PercentSalaryHike_sqrt'].skew()
# transform_data['PercentSalaryHike_sqrt'].kurtosis()
# transform_data['PercentSalaryHike_cbrt'].skew()
# transform_data['PercentSalaryHike_cbrt'].kurtosis()
# transform_data['PercentSalaryHike_box_cox'].skew()
# transform_data['PercentSalaryHike_box_cox'].kurtosis()
# transform_data['PercentSalaryHike_yeo_johnson'].skew()
# transform_data['PercentSalaryHike_yeo_johnson'].kurtosis()
# #"""Best Transformation after comparing skewness
# #NumCompaniesWorked-yeo_johnson,TotalWorkingYears-yeo_johnson,YearsAtCompany-yeo_johnson,DistanceFromHome-yeo_johnson,MonthlyIncome-yeo_johnson,PercentSalaryHike-yeo_johnson

# #now we know which column to use which method so we fit transforming in real data frame now 
# # DistanceFromHome --> Yeo Johnson
pt_distance = PowerTransformer(method="yeo-johnson")
pt_income = PowerTransformer(method="yeo-johnson")
pt_companies = PowerTransformer(method="yeo-johnson")
pt_total_years = PowerTransformer(method="yeo-johnson")
pt_company_years = PowerTransformer(method="yeo-johnson")

employ_df["DistanceFromHome"] = pt_distance.fit_transform(
    employ_df[["DistanceFromHome"]]
)

joblib.dump(pt_distance, "DistanceFromHome.joblib")

employ_df["MonthlyIncome"] = pt_income.fit_transform(
    employ_df[["MonthlyIncome"]]
)

joblib.dump(pt_income, "MonthlyIncome.joblib")

employ_df["NumCompaniesWorked"] = pt_companies.fit_transform(
    employ_df[["NumCompaniesWorked"]]
)

joblib.dump(pt_companies, "NumCompaniesWorked.joblib")

employ_df["TotalWorkingYears"] = pt_total_years.fit_transform(
    employ_df[["TotalWorkingYears"]]
)

joblib.dump(pt_total_years, "total_working_years_power.joblib")

employ_df["YearsAtCompany"] = pt_company_years.fit_transform(
    employ_df[["YearsAtCompany"]]
)

joblib.dump(pt_company_years, "years_at_company_power.joblib")

# #getting all the catagorical columns
# cat_col=employ_df.select_dtypes(include=object)
# cat_col.columns


# #anova test and chisquare test  (whether the column is important) ie compare the column with target column and check whether the column is important
# X = employ_df.drop(['Attrition'],axis=1)
# y = employ_df['Attrition']

# # Separate categorical and numerical columns first
# X_Cat = employ_df.select_dtypes(include="object")
# X_num = employ_df.select_dtypes(include="number")
# X_Cat.drop(['Attrition'],axis=1,inplace=True)

# from sklearn.preprocessing import LabelEncoder
# le1=LabelEncoder()
# y= le1.fit_transform(y)


# X_Cat_encoded = X_Cat.copy()
# for col in X_Cat_encoded.columns:
#     le = LabelEncoder()
#     X_Cat_encoded[col] = le.fit_transform(X_Cat_encoded[col])
    
# #X_Cat_encoded.drop(['Attrition'],axis=1,inplace=True)


# # chi squre test 
# from sklearn.feature_selection import SelectKBest, chi2

# chi = SelectKBest(score_func=chi2, k="all")
# chi.fit(X_Cat_encoded, y)

# chi_result = pd.DataFrame({
#     "Feature": X_Cat_encoded.columns,
#     "Chi2 Score": chi.scores_,
#     "P-Value": chi.pvalues_
# }).sort_values(by="Chi2 Score", ascending=False)

# print(chi_result)
# # if p value is > than 0.05 then column is of no use 

# from sklearn.feature_selection import f_classif

# anova = SelectKBest(score_func=f_classif, k="all")
# anova.fit(X_num, y)

# anova_result = pd.DataFrame({
#     "Feature": X_num.columns,
#     "ANOVA Score": anova.scores_,
#     "P-Value": anova.pvalues_
# }).sort_values(by="ANOVA Score", ascending=False)

# print(anova_result)

##dividing into independent and dependent variables
x_employ = employ_df.drop(['Attrition'],axis=1)
y_employ=employ_df['Attrition']

x_employ=x_employ.drop(['Department','EducationField','Gender','BusinessTravel','YearsSinceLastPromotion','Education','PercentSalaryHike','MonthlyRate','HourlyRate','PerformanceRating'],axis=1)

##label=attrition,buisnesstravel,gender,JobRole,overtime
##one hot =Department,EducationField,,MaritalStatus

##jobrole
x_employ["JobRole"].unique()
def changing_jobrole(x):
    if 'Manager' in x or 'manager' in x:
            return 0
    elif 'director'  in x or 'Director' in x:
        return 1 
    elif 'representative'  in x or 'Representative' in x:
        return 2
    else:
        return 3 
    
    
x_employ["JobRole"]=x_employ["JobRole"].apply(changing_jobrole)

##Attrition
employ_df["Attrition"].unique()
def changing_Attrition(x):
    if 'yes' in x or 'Yes' in x:
            return 0
    else:
        return 1
    
y_employ = y_employ.apply(changing_Attrition)

# #BusinessTravel
# employ_df["BusinessTravel"].unique()
# def changing_BusinessTravel(x):
#     if 'Travel_Frequently' in x:
#             return 0
#     elif 'Travel_Rarely'  in x:
#         return 1 
#     elif 'Non-Travel'  in x:
#         return 2
#     else:
#         return 3 


# employ_df["BusinessTravel"]=employ_df["BusinessTravel"].apply(changing_BusinessTravel)

# #Gender
# employ_df["Gender"].unique()
# def changing_Gender(x):
#     if 'Male' in x :
#             return 0
#     else:
#         return 1


# employ_df["Gender"]=employ_df["Gender"].apply(changing_Gender)


##OverTime
x_employ["OverTime"].unique()
def changing_OverTime(x):
    if 'yes' in x or 'Yes' in x :
            return 0
    else:
        return 1


x_employ["OverTime"]=x_employ["OverTime"].apply(changing_OverTime)


##one hot encoding 
catagorical_columns=['MaritalStatus']
one_hot_encoder=OneHotEncoder(sparse_output=False,handle_unknown='ignore')

preprocessor = ColumnTransformer(
    [('one_hot_encoder',one_hot_encoder,catagorical_columns)],remainder='passthrough')

x_employ=preprocessor.fit_transform(x_employ)


##to save the object ie the file of encoding objects 
joblib.dump(preprocessor,'catagorical_columns_encode.joblib')





#train test split
X_train,X_test,y_train,y_test=train_test_split(x_employ,y_employ,test_size=0.2,random_state=42,stratify=y_employ)#stratify = y_employ is because ye want that equal no of yes and no rows will be picked during splitting

# #seeing splitting of yes and no in attrition before smote 
# plt.bar(y_train.value_counts().index,y_train.value_counts().values,
#           color=['skyblue','salmon'])
# plt.xticks([0,1]),['yes','no']
# plt.ylabel('count')
# plt.title("attrition disytribution before smote")
# plt.show()


# #applying smote to form equal number of yes or no  
# smote=SMOTE(sampling_strategy='minority',random_state=42,k_neighbors=3)
# #fitting in na new object x_train_sm and y_train_sm
# X_train_sm,y_train_sm=smote.fit_resample(X_train,y_train)
# # #seeing if yes or no values of attrition are equal or not
# # plt.bar(y_train_sm.value_counts().index,y_train_sm.value_counts().values,
# #           color=['skyblue','salmon'])
# # plt.xticks([0,1]),['yes','no']
# # plt.ylabel('count')
# # plt.title("attrition disytribution after smote")
# # plt.show()



#savin all the files(encoding,scalar) into one file ie pipeline.with pipeline we train with all the data 
pipeline_d=Pipeline([
    ('model',RandomForestClassifier())
    ])
pipeline_d.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
y_pred=pipeline_d.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)

model=LogisticRegression(C=1.003,l1_ratio=0.005)
model.fit(X_train,y_train)
y_predict=model.predict(X_train)
joblib.dump(pipeline_d, "employ_attrition_pipeline.joblib")

# slope_m=model.coef_
# bais_b=model.intercept_


#confusion matrix
#     po           n1
#p0   236         11     #236 values are yes and model predict yes and 11 yes thi jo no predict kr dii 

#n1   32           16    #16 no thi jo no hi predict kri and 32 no thi jo yes predicct kri and this type of error is called type error and we want that these values will be as low as it can 


#grid search #hyper tunniing



# #applying knn algorithm 

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Create KNN model
# knn = KNeighborsClassifier(n_neighbors=15)#Here 5 is K, meaning KNN looks at the 5 nearest data points.

# # Train the model
# knn.fit(X_train, y_train)

# # Prediction
# y_pred = knn.predict(X_test)

# accuracy=accuracy_score(y_test,y_pred)

# #gridsearch cv -GridSearchCV is a technique used for hyperparameter tuning in machine learning.
# #It tries different combinations of hyperparameter values and finds the combination that gives the best model performance using cross-validation.
# from sklearn.model_selection import GridSearchCV
# param_grid={'n_neighbors':range(1,21)}
# knn=KNeighborsClassifier()
# grid_search=GridSearchCV(knn, param_grid,cv=5)
# grid_search.fit(X_train,y_train)

# optimal_k=grid_search.best_params_['n_neighbors']
# print(f"the optimal k value is :{optimal_k}")


# #applying support vector 
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.svm import SVC
# svm=SVC(kernel='linear')
# svm.fit(X_train,y_train)

# y_pred = svm.predict(X_test)
# accuracy=accuracy_score(y_test,y_pred)
# accuracy=accuracy_score(y_test,y_pred)



# #decision  tree 
# # tree based structure 
# # if classification problem then  majority wins and if regression then avearage wins :
# # it will find through git index that which featured column become root node 




































'''


Here's a concise table covering the transformations from **Log** to **Yeo-Johnson**, including when to use them and the basic Python code.

| Transformation  | Formula                                    | When to Use                                   | Handles Negative Values?       | Base Python Code                         |
| --------------- | ------------------------------------------ | --------------------------------------------- | ------------------------------ | ---------------------------------------- |
| **Log**         | (y=\log(x))                                | Highly right-skewed positive data             | ❌ No                           | `df[col] = np.log(df[col])`              |
| **Log1p**       | (y=\log(1+x))                              | Right-skewed data with zeros                  | ❌ No negatives (zeros allowed) | `df[col] = np.log1p(df[col])`            |
| **Square Root** | (y=\sqrt{x})                               | Moderately right-skewed data                  | ❌ No                           | `df[col] = np.sqrt(df[col])`             |
| **Cube Root**   | (y=\sqrt[3]{x})                            | Moderate skewness, works with negative values | ✅ Yes                          | `df[col] = np.cbrt(df[col])`             |
| **Box-Cox**     | (y=\frac{x^\lambda-1}{\lambda})            | Strongly skewed positive data                 | ❌ No                           | `PowerTransformer(method='box-cox')`     |
| **Yeo-Johnson** | Automatically chooses power transformation | Strongly skewed data with negatives or zeros  | ✅ Yes                          | `PowerTransformer(method='yeo-johnson')` |

## Base Code Examples

### 1. Log Transformation

```python
import numpy as np

df["Feature"] = np.log(df["Feature"])
```

---

### 2. Log1p Transformation (Recommended if zeros exist)

```python
import numpy as np

df["Feature"] = np.log1p(df["Feature"])
```

---

### 3. Square Root Transformation

```python
import numpy as np

df["Feature"] = np.sqrt(df["Feature"])
```

---

### 4. Cube Root Transformation

```python
import numpy as np

df["Feature"] = np.cbrt(df["Feature"])
```


---

### 6. Box-Cox Transformation

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method="box-cox")

df["Feature"] = pt.fit_transform(df[["Feature"]])
```

> **Note:** All values in `Feature` must be **strictly greater than 0**.

---

### 7. Yeo-Johnson Transformation

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method="yeo-johnson")

df["Feature"] = pt.fit_transform(df[["Feature"]])
```

> **Note:** Yeo-Johnson works with **negative values, zero, and positive values**.

---

## Quick Selection Guide

| Data Condition                            | Recommended Transformation   |
| ----------------------------------------- | ---------------------------- |
| Positive values, highly right-skewed      | `np.log()`                   |
| Positive values including zeros           | `np.log1p()`                 |
| Moderately right-skewed                   | `np.sqrt()`                  |
| Contains negative values                  | `np.cbrt()` or `Yeo-Johnson` |
| Strongly skewed positive values           | `Box-Cox`                    |
| Strongly skewed with negative/zero values | `Yeo-Johnson`                |
| Very large values need compression (rare) | `Reciprocal`                 |
'''











