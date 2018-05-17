---
layout: post
title: "Exploratory Data Analysis"
author: "Richard Penfil"
categories: Algorithm
tags: [Decision Tree, Machine Learning]
image: ledingclub_pic.jpg
---

```python
#Import Modules
import os
import numpy as np
import pandas as pd
import datetime 
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from utils import (checking_na, now, proc_import)
```


```python
#Import Data
df = proc_import("accepted_2007_to_2017.csv")
```

    2018-04-16 18:33:54: Reading accepted_2007_to_2017.csv.
    2018-04-16 18:33:54: The data contains 1646801 observations with 150 columns
    

# EDA & Cleaning


```python
#Export Fields (/%Null) and sample
checking_na(df).to_csv("missing.csv")
df.head(5).to_csv("head.csv")
df.describe().to_csv("stats.csv")
df.dtypes.to_csv("types.csv")
```


```python
#proc_hist(application_type)
fig = plt.figure(figsize=(5,5))
ax = sns.countplot(df.loan_status)
ax.set_title("Count of Loan Status")
plt.xticks(rotation='vertical')
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (
            p.get_x(), p.get_height()*1.01))
plt.show()
```


![alt text](https://github.com/rp4/rp4.github.io/blob/master/assets/img/EDA_files/EDA_1.png)



```python
#Frequencies of Objects
print(df.term.value_counts(dropna=False))
print(df.disbursement_method.value_counts(dropna=False))
print(df.application_type.value_counts(dropna=False))
print(df.grade.value_counts(dropna=False))
print(df.sub_grade.value_counts(dropna=False))
print(df.emp_length.value_counts(dropna=False))
print(df.home_ownership.value_counts(dropna=False))
print(df.pymnt_plan.value_counts(dropna=False))
print(df.addr_state.value_counts(dropna=False))
print(df.verification_status.value_counts(dropna=False))
print(df.purpose.value_counts(dropna=False))
```

     36 months    1182406
     60 months     464372
    NaN                23
    Name: term, dtype: int64
    Cash         1643597
    DirectPay       3181
    NaN               23
    Name: disbursement_method, dtype: int64
    Individual    1612264
    Joint App       34514
    NaN                23
    Name: application_type, dtype: int64
    C      490784
    B      485360
    A      272724
    D      237512
    E      111961
    F       37411
    G       11026
    NaN        23
    Name: grade, dtype: int64
    C1     110224
    B5     103448
    B4     102535
    B3     100516
    C2      99275
    C4      96807
    C3      96431
    B2      89461
    B1      89400
    C5      88047
    A5      75310
    D1      61295
    A4      58472
    A1      54265
    D2      53574
    D3      47149
    A2      42347
    A3      42330
    D4      41270
    D5      34224
    E1      28924
    E2      25506
    E3      21831
    E4      18606
    E5      17094
    F1      11428
    F2       8537
    F3       7174
    F4       5649
    F5       4623
    G1       3359
    G2       2550
    G3       1991
    G4       1629
    G5       1497
    NaN        23
    Name: sub_grade, dtype: int64
    10+ years    549538
    2 years      148367
    < 1 year     133332
    3 years      130871
    1 year       107680
    5 years      101848
    4 years       98103
    NaN           95221
    6 years       75568
    8 years       72664
    7 years       70395
    9 years       63214
    Name: emp_length, dtype: int64
    MORTGAGE    815170
    RENT        653236
    OWN         177629
    ANY            507
    OTHER          182
    NONE            54
    NaN             23
    Name: home_ownership, dtype: int64
    n      1645863
    y          915
    NaN         23
    Name: pymnt_plan, dtype: int64
    CA     230904
    NY     137063
    TX     135564
    FL     115714
    IL      66540
    NJ      61088
    PA      57032
    OH      55414
    GA      53916
    VA      46927
    NC      45955
    MI      43098
    MD      39111
    AZ      38663
    MA      37906
    CO      34448
    WA      34424
    MN      29409
    IN      27096
    MO      26291
    CT      25849
    TN      25415
    NV      23320
    WI      21722
    AL      20337
    SC      20114
    OR      19403
    LA      19245
    KY      15967
    OK      15044
    KS      14173
    AR      12382
    UT      11034
    NM       8856
    MS       8752
    NH       8092
    HI       8055
    RI       7263
    WV       5156
    NE       4856
    DE       4665
    MT       4608
    DC       4122
    AK       3934
    WY       3566
    VT       3513
    SD       3316
    ME       2982
    ND       2282
    ID       2178
    NaN        23
    IA         14
    Name: addr_state, dtype: int64
    Source Verified    628245
    Not Verified       516009
    Verified           502524
    NaN                    23
    Name: verification_status, dtype: int64
    debt_consolidation    955783
    credit_card           363962
    home_improvement      109031
    other                  93576
    major_purchase         35596
    medical                18901
    small_business         18613
    car                    17641
    moving                 11388
    vacation               11152
    house                   7268
    wedding                 2350
    renewable_energy        1094
    educational              423
    NaN                       23
    Name: purpose, dtype: int64
    


```python
#Keep Individuals only
df=df.loc[df['application_type'].isin(["Individual"])]
#Keep CO, Paid, and Current
df=df.loc[df['loan_status'].isin(["Fully Paid","Charged Off","Current"])]
#Change to Binary
df['CO'] = df['loan_status'].str.contains("Charged").astype(int)
```


```python
#Change Grade to Number
grades = {'grade':df.grade.unique()}
grades = pd.DataFrame (data=grades)
grades=grades.sort_values('grade')
grades.reset_index(level=0, inplace=True)
grades['grd'] = grades.index +1
#Left Join
df = pd.merge(df, grades, how='left', on=['grade'])
df['grd'].isnull().sum()

```




    0




```python
#Change Sub_Grade to Number
subgrades = {'sub_grade':df.sub_grade.unique()}
subgrades = pd.DataFrame (data=subgrades)
subgrades=subgrades.sort_values('sub_grade')
subgrades.reset_index(level=0, inplace=True)
subgrades['sub_grd'] = subgrades.index +1
#Left Join
df = pd.merge(df, subgrades, how='left', on=['sub_grade'])
df['grd'].isnull().sum()
```




    0




```python
#Change Eployment Years to Number
emp_years = {'emp_length':df.emp_length.unique(),
             'emp_years' : [10,8,6,0,2,9,7,5,3,1,0,4]}
emp_years= pd.DataFrame (data=emp_years)
#Left Join
df = pd.merge(df, emp_years, how='left', on=['emp_length'])
df['emp_years'].isnull().sum()
```




    0




```python
#Change to Date Type
df['issue_d']=pd.to_datetime(df.issue_d, errors = 'ignore')
```


```python
#Create Dummy Variables
dummies = pd.get_dummies(df['home_ownership']).rename(
    columns=lambda x: 'is_' + str(x))
df = pd.concat([df, dummies], axis=1)

dummies = pd.get_dummies(df['verification_status']).rename(
    columns=lambda x: 'is_' + str(x))
df = pd.concat([df, dummies], axis=1)

dummies = pd.get_dummies(df['term']).rename(
    columns=lambda x: 'is_' + str(x))
df = pd.concat([df, dummies], axis=1)

dummies = pd.get_dummies(df['disbursement_method']).rename(
    columns=lambda x: 'is_' + str(x))
df = pd.concat([df, dummies], axis=1)

```


```python
#double check types
df.dtypes.to_csv("types.csv")
```


```python
#Charge Off by Grade
f = {'CO':['sum','count','mean']}
CO_x_grade = df.groupby('grd').agg(f)

# plot histogram and approval rate
fig, host = plt.subplots()
styles1 = ['rs-']
CO_x_grade[list(CO_x_grade)[2]].plot(secondary_y=True, 
                                     marker='.', style=styles1)
CO_x_grade[list(CO_x_grade)[1]].plot(kind='bar',lw=2,colormap='jet',
                        title='Charge offs by Grade')
plt.ylabel("Charge off rate")
```




    <matplotlib.text.Text at 0x14d1090af28>




![png](EDA_files/EDA_13_1.png)


As expected the riskier grades have higher charge off rates. We also see B grades and C grades are the most common.


```python
#Interest Rate by Grade
f = {'int_rate':['sum','count','mean']}
Int_x_grade = df.groupby('sub_grade').agg(f)

# plot histogram and approval rate
fig, host = plt.subplots()
styles1 = ['rs-']
Int_x_grade[list(Int_x_grade)[2]].plot(kind='bar',lw=2,colormap='jet',
                        title='Interest rate by Sub_Grade')
plt.ylabel("Interest Rate")
```




    <matplotlib.text.Text at 0x1dd109d09b0>




![png](EDA_files/EDA_15_1.png)


Again, as expected the avg. interest rate is higher for riskier tiers, to accomodate for the higher charge off rates


```python
#Funded amt by Purpose
f = {'funded_amnt':['sum','count','mean']}
amt_x_purpose = df.groupby('purpose').agg(f)
amt_x_purpose.sort_values(list(amt_x_purpose)[1],ascending=False)
# plot histogram and avg loan size
fig, host = plt.subplots()
styles1 = ['rs-']
amt_x_purpose[list(amt_x_purpose)[2]].plot(secondary_y=True, 
                                           marker='.', style=styles1)
amt_x_purpose[list(amt_x_purpose)[1]].plot(kind='bar',lw=2,colormap='jet',
                        title='Loans by Purpose and Avg. Loan Size')
plt.ylabel("Avg Loan Size")
```




    <matplotlib.text.Text at 0x14cb1575940>




![png](EDA_files/EDA_17_1.png)


Debt Consolidation and credit cards make up the majority of lending club loans


```python
ct=pd.crosstab([df.issue_d], [df.grd], values=df.CO, aggfunc=[np.mean])
plt.figure(); ct.plot();
```


    <matplotlib.figure.Figure at 0x12f0b4b1b70>



![png](EDA_files/EDA_19_1.png)


The wild deviations in 2008 - 2010 can be attributed the small sample size. Also, the decreasing rates in 2017 are of course due to the immature loans


```python
#Create Index of Fields to be graphed
preds = df.ix[:,0:111].columns

#Format output
plt.figure(figsize=(12,111*4))
gs = gridspec.GridSpec(111, 1)

#Create overlapping histogram for each variable
for i, cn in enumerate(df[preds]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[cn][df.CO == 1], bins=50)
    sns.distplot(df[cn][df.CO == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()
```


![png](EDA_files/EDA_21_0.png)



```python
#Drop extra fields and trap dummy variables

extra = ['is_OWN','is_Verified','grade','is_DirectPay',
'is_ 60 months','pymnt_plan','home_ownership','addr_state',
'verification_status','id','member_id','desc','title','zip_code',
'sec_app_fico_range_low','sec_app_fico_range_high',
'sec_app_earliest_cr_line','sec_app_inq_last_6mths',
'sec_app_mort_acc','sec_app_open_acc',
'sec_app_revol_util','sec_app_open_act_il',
'sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths',
'sec_app_collections_12_mths_ex_med',
'sec_app_mths_since_last_major_derog',
'loan_status','term',
'sub_grade','emp_title','emp_length',
'earliest_cr_line','initial_list_status',
'last_pymnt_d','next_pymnt_d','last_credit_pull_d',
'application_type','verification_status_joint','hardship_flag',
'hardship_type','hardship_reason','hardship_status','hardship_start_date',
'hardship_end_date','payment_plan_start_date','hardship_loan_status',
'disbursement_method','debt_settlement_flag','debt_settlement_flag_date',
'settlement_status','settlement_date','dti_joint',
'revol_bal_joint','annual_inc_joint']

df = df.drop(extra,  axis=1)
```


```python
#Drop Performance Variables
df = df.drop(['collections_12_mths_ex_med'
,'policy_code','max_bal_bc','acc_now_delinq'
,'tot_coll_amt','delinq_amnt','num_tl_120dpd_2m'
,'num_tl_30dpd','num_tl_90g_dpd_24m','hardship_dpd'
,'hardship_payoff_balance_amount','hardship_last_payment_amount'
,'settlement_amount','settlement_percentage'
,'settlement_term','collection_recovery_fee'
,'recoveries','last_fico_range_low','out_prncp'
,'last_pymnt_amnt','installment','total_rec_late_fee'
], axis=1)
```

# Pre-Processing


```python
# Seperate Predictors from Dependent
X = df.drop(['CO','issue_d','purpose'], axis=1)
y = df.CO
checking_na(df).to_csv("missing.csv")
```


```python
#Keep Column Names
headers=list(X)
```


```python
# Create our imputer to replace missing values with the mean.
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = imp.fit(X)
X=imp.transform(X)
X = pd.DataFrame(X)
```


```python
#Standardize Columns for speedier model building
X=X.values #returns numpy array
stan_scalar = preprocessing.StandardScaler()
x_scaled = stan_scalar.fit_transform(X)
df = pd.DataFrame(x_scaled)
```


```python
#Restore Original Column Names
df.columns = headers
y.columns = ['CO']
```


```python
df = df.join(y)
df.to_csv('Lending_club_clean.csv')
```


```python
#Split Training & Test
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['CO'], axis=1), df.CO, test_size=0.01, random_state=0)
```


```python
X_train.to_csv('X_train.csv')
X_test.to_csv('X_test.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')
```

#### Remark:

The above work is only a part of a complete analysis. Much more can be done in terms of EDA, cleaning, feature engineering, and preprocessing techniques.


```python

```
