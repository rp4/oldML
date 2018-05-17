---
layout: post
title: "Gradient Boosting"
author: "Richard Penfil"
categories: Algorithm
tags: [Gradient Boosting, Machine Learning]
image: gradient_boosting_head.jpg
---

#### Below is an example of a Gradient Boosting model. I used a loan dataset (found here:https://www.kaggle.com/zhijinzhai/loandata), which contains loans that customers paid off or neglected to pay. The model created attempts to quantify the risk of each customer.


```python
#Import Data
df = proc_import("Loan payments data.csv")
print(checking_na(df))
```

    2018-05-15 09:56:26: Reading Loan payments data.csv.
    2018-05-15 09:56:26: The data contains 500 observations with 11 columns
                  df_bool  df_amt  missing_ratio_percent
    paid_off_time    True     100                   20.0
    past_due_days    True     300                   60.0
    

### Contents


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 500 entries, 0 to 499
    Data columns (total 11 columns):
    Loan_ID           500 non-null object
    loan_status       500 non-null object
    Principal         500 non-null int64
    terms             500 non-null int64
    effective_date    500 non-null object
    due_date          500 non-null object
    paid_off_time     400 non-null object
    past_due_days     200 non-null float64
    age               500 non-null int64
    education         500 non-null object
    Gender            500 non-null object
    dtypes: float64(1), int64(3), object(7)
    memory usage: 43.0+ KB
    

## Clean Data/Prep for Model

### Dependent Variable: Loan Status


```python
print(df.loan_status.value_counts(dropna=False))
fig = plt.figure(figsize=(5,5))
ax = sns.countplot(df.loan_status)
ax.set_title("Count of Loan Status")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (
            p.get_x(), p.get_height()*1.01))
plt.show()
```

    PAIDOFF               300
    COLLECTION            100
    COLLECTION_PAIDOFF    100
    Name: loan_status, dtype: int64
    

<a href="https://github.com/rp4/rp4.github.io/blob/master/assets/img/GB_files/Gradient_Boost_1.png"><img src="{{ site.github.url }}/assets/img/GB_files/Gradient_Boost_1.png"></a>



```python
#Change to Binary
df['loan_status'] = df['loan_status'].str.contains("COLLECTION").astype(int)
```

### Predictor Variables


```python
# Drop Unnecessary fields
df = df.drop(['Loan_ID', 'effective_date', 'due_date', 
              'paid_off_time', 'past_due_days'], axis=1)
```


```python
#Create Dummy Variables
dummies = pd.get_dummies(df['education']).rename(
    columns=lambda x: 'is_' + str(x))
df = pd.concat([df, dummies], axis=1)
df = df.drop(['education'],  axis=1)

dummies = pd.get_dummies(df['Gender']).rename(
    columns=lambda x: 'is_' + str(x))
df = pd.concat([df, dummies], axis=1)
df = df.drop(['Gender'], axis=1)

dummy_var = ['is_female', 'is_Master or Above']
df = df.drop(dummy_var, axis = 1)
```


```python
# Seperate Predictors from Dependent
X = df.drop(['loan_status'], axis=1)
y = df.loan_status
```

# Gradient Boost  Model


```python
#Split Training & Test
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['loan_status'], axis=1), df.loan_status, 
    test_size=0.33, random_state=0)
```


```python
#Sci-Kit Learn Gradient Boost Model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01}
GB = GradientBoostingClassifier(**params)

GB.fit(X_train, y_train)

mse = mean_squared_error(y_test, GB.predict(X_test))
print("MSE: %.4f" % mse)
```

    MSE: 0.4303
    

The Gradient Boosting Model has a few hyperparameters including the learning rate, the number of braches each tree can have, the maximum depth of each tree, and the number of trees to run.


```python
# Plot feature importance
feature_importance = GB.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (
    feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
```

<a href="https://github.com/rp4/rp4.github.io/blob/master/assets/img/GB_files/Gradient_Boost_2.png"><img src="{{ site.github.url }}/assets/img/GB_files/Gradient_Boost_2.png"></a>


The Gradient Boosting model provides feature importance based on the amount of times each variable is used and the lift given from each variable. In this model, age appears as the most important variable, while bachelor status appears as the least important.


```python
    #PARTIAL DEPENDENCY PLOTS
print('Convenience plot with ``partial_dependence_plots``')
#features = [0, 5, 1, 2, (5, 1)]
features = [0,1,2,6,(1,0)]
fig, axs = plot_partial_dependence(GB, X_train, features,
                                   feature_names=X_train.columns,
                                   n_jobs=3, grid_resolution=50)
fig.suptitle('Partial dependence of Charge Off\n'
             'for the Loan dataset')
plt.subplots_adjust(top=0.9)
```

    Convenience plot with ``partial_dependence_plots``
    


<a href="https://github.com/rp4/rp4.github.io/blob/master/assets/img/GB_files/Gradient_Boost_3.png"><img src="{{ site.github.url }}/assets/img/GB_files/Gradient_Boost_3.png"></a>


Another bi-product of the Gradient Boosted Model is the partial dependence plot.
Partial dependence plots show the dependence between the target function and the set of features, marginalizing over the values of all other features. As seen above, lower term has a linear relationship with Chargeoffs until term is equal to 15. Above 15 the term does not seem to contribute to chargeoffs.


```python
# Plot training deviance
# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(GB.staged_predict(X_test)):
    test_score[i] = GB.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, GB.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
```




    <matplotlib.text.Text at 0x1ed8c70d4e0>




<a href="https://github.com/rp4/rp4.github.io/blob/master/assets/img/GB_files/Gradient_Boost_4.png"><img src="{{ site.github.url }}/assets/img/GB_files/Gradient_Boost_4.png"></a>


While the Training Set Deviance decreases as the boosting iterations increase, the Test Set Deviance slightly increases. This suggests that the model is overfitting to the Training data with the additional iterations.


```python
#XGBoost
XGB = XGBClassifier()
XGB.fit(X_train, y_train)
# make predictions for test data
xgb_pred = XGB.predict(X_test)
xgb_class = [round(value) for value in xgb_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, xgb_class)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

    Accuracy: 55.15%
    

Above is an example of XGBoost. While it is similar to sci-kit learns Gradient Boosting Model, it has the capability to run out of core on distributed GPU's and in general is faster on larger datasets.


```python
#Confusion Matrix
cnf_matrix = confusion_matrix(y_test,xgb_pred)
np.set_printoptions(precision=2)

ax = sns.heatmap(cnf_matrix, annot=True, fmt="d",
                 cmap="YlGnBu",cbar=False)
plt.ylabel('True label')
plt.xlabel('Predicted label')
```




    <matplotlib.text.Text at 0x1ed8e106080>




<a href="https://github.com/rp4/rp4.github.io/blob/master/assets/img/GB_files/Gradient_Boost_5.png"><img src="{{ site.github.url }}/assets/img/GB_files/Gradient_Boost_5.png"></a>



```python

```
