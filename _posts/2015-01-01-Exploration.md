---
layout: post
title: "Exploration"
author: "Richard Penfil"
categories: documentation
tags: [documentation,sample]
image: visa-1.jpg
---

## U.S. Permanent Visa Applications
Below is a simple initial exploration of a dataset of containing US permanent Visa Applications (which can be found here: https://www.kaggle.com/jboysen/us-perm-visas)
While every dataset is different, and will need individualized massaging, these are some initial steps to start to understand and clean a dataset.


```python
# import packages
import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sqlite3
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
```


```python
df = pd.read_csv('us_perm_visas.csv', low_memory=False, thousands=',')
#df.describe
#df.isnull().sum()
```

### CLEAN DATA
First I will clean the data by dropping uneccesary columns,
combining the country_of_citzenship with country_of_citizenship column 
and case_number with case_no


```python
#Keep selected columns
df=df[['case_number', 'case_no', 'case_received_date', 'case_status', 'decision_date', 
              'employer_city', 'employer_state','country_of_citzenship','country_of_citizenship','class_of_admission',
               'wage_offer_from_9089','wage_offer_to_9089',
               'wage_offered_from_9089','wage_offered_to_9089',
                'wage_offer_unit_of_pay_9089','wage_offered_unit_of_pay_9089']]
#Coalesce
df['case_id'] = df.case_number.combine_first(df.case_no)
df['orig_country'] = df.country_of_citzenship.combine_first(df.country_of_citizenship)
df = df.drop(['country_of_citizenship', 'country_of_citzenship','case_no','case_number'], axis=1)
df.head()
```

    C:\Users\user0\AppData\Local\Continuum\Anaconda3\lib\site-packages\ipykernel\__main__.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    C:\Users\user0\AppData\Local\Continuum\Anaconda3\lib\site-packages\ipykernel\__main__.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>case_received_date</th>
      <th>case_status</th>
      <th>decision_date</th>
      <th>employer_city</th>
      <th>employer_state</th>
      <th>class_of_admission</th>
      <th>wage_offer_from_9089</th>
      <th>wage_offer_to_9089</th>
      <th>wage_offered_from_9089</th>
      <th>wage_offered_to_9089</th>
      <th>wage_offer_unit_of_pay_9089</th>
      <th>wage_offered_unit_of_pay_9089</th>
      <th>case_id</th>
      <th>orig_country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Certified</td>
      <td>2012-02-01</td>
      <td>NEW YORK</td>
      <td>NY</td>
      <td>J-1</td>
      <td>75629.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yr</td>
      <td>NaN</td>
      <td>A-07323-97014</td>
      <td>ARMENIA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Denied</td>
      <td>2011-12-21</td>
      <td>CARLSTADT</td>
      <td>NY</td>
      <td>B-2</td>
      <td>37024.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yr</td>
      <td>NaN</td>
      <td>A-07332-99439</td>
      <td>POLAND</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Certified</td>
      <td>2011-12-01</td>
      <td>GLEN ALLEN</td>
      <td>VA</td>
      <td>H-1B</td>
      <td>47923.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yr</td>
      <td>NaN</td>
      <td>A-07333-99643</td>
      <td>INDIA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Certified</td>
      <td>2011-12-01</td>
      <td>FLUSHING</td>
      <td>NY</td>
      <td>B-2</td>
      <td>10.97</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>hr</td>
      <td>NaN</td>
      <td>A-07339-01930</td>
      <td>SOUTH KOREA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Certified</td>
      <td>2012-01-26</td>
      <td>ALBANY</td>
      <td>NY</td>
      <td>L-1</td>
      <td>100000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yr</td>
      <td>NaN</td>
      <td>A-07345-03565</td>
      <td>CANADA</td>
    </tr>
  </tbody>
</table>
</div>



#### Next, change dependent variable to integer


```python
df['Approved'] = df['case_status'].str.contains("Certified").astype(int)
```

#### Frequencies and Approval Rates by Continent


```python
#Left Join to continent data
df2 = pd.read_csv('continents.csv')
df = pd.merge(df, df2, how='left', on=['orig_country'])
df['Continent'].isnull().sum()
```




    59




```python
#Aggregate by Country of continent
f = {'Approved':['sum','count','mean']}
df_group = df.groupby('Continent').agg(f)

#Sort by count of application for each country of origins
df_group.sort_values(list(df_group)[1],ascending=False)

#Change count to distribution percent
#total_count = df.shape[0]
#df_group[list(df_group)[1]] /= total_count
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">Approved</th>
    </tr>
    <tr>
      <th></th>
      <th>sum</th>
      <th>count</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>Continent</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Asia</th>
      <td>269992</td>
      <td>302645</td>
      <td>0.892108</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>21794</td>
      <td>24924</td>
      <td>0.874418</td>
    </tr>
    <tr>
      <th>North America</th>
      <td>20244</td>
      <td>24442</td>
      <td>0.828246</td>
    </tr>
    <tr>
      <th>South America</th>
      <td>9311</td>
      <td>11249</td>
      <td>0.827718</td>
    </tr>
    <tr>
      <th>Africa</th>
      <td>4565</td>
      <td>5296</td>
      <td>0.861971</td>
    </tr>
    <tr>
      <th>Central America</th>
      <td>2747</td>
      <td>3655</td>
      <td>0.751573</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>1865</td>
      <td>2092</td>
      <td>0.891491</td>
    </tr>
  </tbody>
</table>
</div>



#### Normalize the Salaries to yearly pay


```python
#Create multiplier table
Multiplyer = pd.DataFrame(pd.unique(df[['wage_offer_unit_of_pay_9089','wage_offered_unit_of_pay_9089']].values.ravel('K')))
Multiplyer['multiple'] = (1,2080,12,52,26,0,1,2080,52,12,26)
Multiplyer

#Change select columns to floats
cols = ['wage_offer_from_9089','wage_offer_to_9089','wage_offered_from_9089','wage_offered_to_9089']
df[cols] = df[cols].apply(lambda x: pd.to_numeric(x.astype(str)
                                                   .str.replace(',',''), errors='coerce'))
```


```python
# Join multiplier to dataframe
Multiplyer.columns = ['wage_offer_unit_of_pay_9089', 'offer_multiplyer']
df = pd.merge(df, Multiplyer, how='left', on=['wage_offer_unit_of_pay_9089'])

Multiplyer.columns = ['wage_offered_unit_of_pay_9089', 'offered_multiplyer']
df = pd.merge(df, Multiplyer, how='left', on=['wage_offered_unit_of_pay_9089'])

# Multiply Columns
df['wage_offer_from_9089']=df['wage_offer_from_9089']*df['offer_multiplyer']
df['wage_offer_to_9089']=df['wage_offer_to_9089']*df['offer_multiplyer']
df['wage_offered_from_9089']=df['wage_offered_from_9089']*df['offered_multiplyer']
df['wage_offered_to_9089']=df['wage_offered_to_9089']*df['offered_multiplyer']
```

## PLOT DATA


```python
# plot histogram and approval rate
fig, host = plt.subplots()
styles1 = ['rs-']
df_group[list(df_group)[2]].plot(secondary_y=True, marker='.', style=styles1)
df_group[list(df_group)[1]].plot(kind='bar',lw=2,colormap='jet',
                        title='Applications and Approval Rate by Continent')
plt.xticks(rotation='vertical')
plt.ylabel("Approval %")
```




    <matplotlib.text.Text at 0x20bbb0c9eb8>



<a><img src="{{ site.github.url }}/assets/img/exploration_files/exploration_1.jpg"></a>


## Histograms (Approved vs Decline)


```python
#Create Index of Fields to be graphed
salaries = df.ix[:,6:10].columns

#Format output
plt.figure(figsize=(12,4*4))
gs = gridspec.GridSpec(4, 1)

#Create overlapping histogram for each variable
for i, cn in enumerate(df[salaries]):
    df[salaries].dropna()
    ax = plt.subplot(gs[i])
    subset=df[[cn,'Approved']].dropna(axis=0, how='any')
    subset=df[[cn,'Approved']][df[cn]<250000]
    sns.distplot(subset[cn][subset.Approved == 1], bins=50)
    sns.distplot(subset[cn][subset.Approved == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()
```

    C:\Users\user0\AppData\Local\Continuum\Anaconda3\lib\site-packages\statsmodels\nonparametric\kdetools.py:20: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j
    


<a href="https://rp4.github.io/exploration_2/"><img src="{{ site.github.url }}/assets/img/exploration_files/exploration_2.jpg"></a>



```python

```
