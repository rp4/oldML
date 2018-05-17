---
layout: post
title: "Random Forest"
author: "Richard Penfil"
categories: Algorithm
tags: [Random Forest, Machine Learning]
image: Forest_pic.jpg
---

#### Below is an example of a Random Forest model. I used dataset (found here:https://www.kaggle.com/mlg-ulb/creditcardfraud/data), which contains credit card transactions over a two day period. The model created attempts to identify fraudulent transactions based on the provided (masked) attributes.

# Quick and Simple EDA


```python
#Import Data
df = proc_import("creditcard.csv")
```

    2018-05-15 07:30:48: Reading creditcard.csv.
    2018-05-15 07:30:48: The data contains 284807 observations with 31 columns
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
    Time      284807 non-null float64
    V1        284807 non-null float64
    V2        284807 non-null float64
    V3        284807 non-null float64
    V4        284807 non-null float64
    V5        284807 non-null float64
    V6        284807 non-null float64
    V7        284807 non-null float64
    V8        284807 non-null float64
    V9        284807 non-null float64
    V10       284807 non-null float64
    V11       284807 non-null float64
    V12       284807 non-null float64
    V13       284807 non-null float64
    V14       284807 non-null float64
    V15       284807 non-null float64
    V16       284807 non-null float64
    V17       284807 non-null float64
    V18       284807 non-null float64
    V19       284807 non-null float64
    V20       284807 non-null float64
    V21       284807 non-null float64
    V22       284807 non-null float64
    V23       284807 non-null float64
    V24       284807 non-null float64
    V25       284807 non-null float64
    V26       284807 non-null float64
    V27       284807 non-null float64
    V28       284807 non-null float64
    Amount    284807 non-null float64
    Class     284807 non-null int64
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB
    


```python
print(df.Class.value_counts(dropna=False))
fig = plt.figure(figsize=(5,5))
ax = sns.countplot(df.Class)
ax.set_title("Count of Loan Status")
for p in ax.patches:
    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
plt.show()
```

    0    284315
    1       492
    Name: Class, dtype: int64
    


![png](Random_Forest_files/Random_Forest_4_1.png)


From above, we see that the data is extremely unbalanced with only 0.172% fraudulent transactions. 
Usually I would use oversampling and then adjust the results, however here I choose to preserve the unbalanced data as is

### Correlations


```python
# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))


# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x219c25bdc50>




![png](Random_Forest_files/Random_Forest_7_1.png)



```python
#Split Training & Test
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(['Class'], axis=1), df.Class, test_size=0.33, random_state=0)
```

# Random Forest Model


```python
def eval_model_classifier(model, data, target, split_ratio):
    trainX, testX, trainY, testY = train_test_split(data, target, train_size=split_ratio, random_state=0)
    model.fit(trainX, trainY)    
    return model.score(testX,testY)
```


```python
num_estimators_array = np.array([1,5,10,30]) 
num_smpl = 5
num_grid = len(num_estimators_array)
score_array_mu = np.zeros(num_grid)
score_array_sigma = np.zeros(num_grid) 
j=0

print("{}: RandomForestClassification Starts!".format(now()))
for n_estimators in num_estimators_array:
    score_array = np.zeros(num_smpl) # Initialize
    for i in range(0,num_smpl):
        rf_class = RandomForestClassifier(n_estimators = n_estimators, 
                                          n_jobs=1, criterion="gini")
        score_array[i] = eval_model_classifier(rf_class, X_train, y_train, 0.8)
        print("{}: Try {} with n_estimators = {} and score = {}".format(now(), 
                                                                        i, n_estimators, score_array[i]))
    score_array_mu[j], score_array_sigma[j] = np.mean(score_array), np.std(score_array)
    j=j+1

print("{}: RandomForestClassification Done!".format(now()))
```

    2018-05-13 17:02:35: RandomForestClassification Starts!
    2018-05-13 17:02:37: Try 0 with n_estimators = 1 and score = 0.9988118882110938
    2018-05-13 17:02:39: Try 1 with n_estimators = 1 and score = 0.9988296212228684
    2018-05-13 17:02:41: Try 2 with n_estimators = 1 and score = 0.9988473542346432
    2018-05-13 17:02:43: Try 3 with n_estimators = 1 and score = 0.9990246843523904
    2018-05-13 17:02:45: Try 4 with n_estimators = 1 and score = 0.9989714853170663
    2018-05-13 17:02:56: Try 0 with n_estimators = 5 and score = 0.9995034756703078
    2018-05-13 17:03:06: Try 1 with n_estimators = 5 and score = 0.9992906795290112
    2018-05-13 17:03:15: Try 2 with n_estimators = 5 and score = 0.9993438785643354
    2018-05-13 17:03:24: Try 3 with n_estimators = 5 and score = 0.9993793445878848
    2018-05-13 17:03:32: Try 4 with n_estimators = 5 and score = 0.9995212086820826
    2018-05-13 17:03:52: Try 0 with n_estimators = 10 and score = 0.9994857426585331
    2018-05-13 17:04:12: Try 1 with n_estimators = 10 and score = 0.9994148106114342
    2018-05-13 17:04:28: Try 2 with n_estimators = 10 and score = 0.9994502766349836
    2018-05-13 17:04:45: Try 3 with n_estimators = 10 and score = 0.9993970775996596
    2018-05-13 17:05:03: Try 4 with n_estimators = 10 and score = 0.999432543623209
    2018-05-13 17:05:56: Try 0 with n_estimators = 30 and score = 0.9994148106114342
    2018-05-13 17:06:53: Try 1 with n_estimators = 30 and score = 0.9994680096467584
    2018-05-13 17:07:50: Try 2 with n_estimators = 30 and score = 0.9993793445878848
    2018-05-13 17:08:45: Try 3 with n_estimators = 30 and score = 0.9993970775996596
    2018-05-13 17:09:40: Try 4 with n_estimators = 30 and score = 0.9994502766349836
    2018-05-13 17:09:40: RandomForestClassification Done!
    


```python
fig = plt.figure(figsize=(7,3))
plt.errorbar(num_estimators_array, score_array_mu, 
             yerr=score_array_sigma, fmt='k.-')
plt.xscale("log")
plt.xlabel("number of estimators",size = 16)
plt.ylabel("accuracy",size = 16)
plt.xlim(0.9,30)
plt.ylim(0.998,1)
plt.title("Random Forest Classifier", size = 18)
plt.grid(which="both")
plt.show();
```


![png](Random_Forest_files/Random_Forest_12_0.png)



```python
#Select Hyperparameters
forest = RandomForestClassifier(n_jobs=2,n_estimators=31)
forest.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=31, n_jobs=2,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)



#### Feature Importance


```python
importances = forest.feature_importances_
indices = np.argsort(importances)
top20 = indices[len(indices)-25:]
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(top20)), importances[top20],color='b',align='center')
plt.yticks(range(len(top20)), X_train[top20])
plt.xlabel('Relative Importance')
```




    <matplotlib.text.Text at 0x219c36ddcf8>




![png](Random_Forest_files/Random_Forest_15_1.png)



```python
select = sklearn.feature_selection.SelectKBest(k=10)
rf = sklearn.ensemble.RandomForestClassifier()

steps = [('feature_selection', select),
        ('random_forest', rf)]

pipeline = sklearn.pipeline.Pipeline(steps)

# fit your pipeline on training
pipeline.fit( X_train, y_train )
# predictions
rf_preds = pipeline.predict( X_test )
# test the predictions
report = sklearn.metrics.classification_report(y_test, rf_preds)
print(report)
```

                 precision    recall  f1-score   support
    
              0       1.00      1.00      1.00     93825
              1       0.92      0.75      0.83       162
    
    avg / total       1.00      1.00      1.00     93987
    
    

#### Confustion Matrix


```python
cnf_matrix = confusion_matrix(y_test,rf_preds)
np.set_printoptions(precision=2)

ax = sns.heatmap(cnf_matrix, annot=True, fmt="d", 
                 cmap="YlGnBu",cbar=False)
plt.ylabel('True label')
plt.xlabel('Predicted label')
```




    <matplotlib.text.Text at 0x1de82f64710>




![png](Random_Forest_files/Random_Forest_18_1.png)


#### ROC Curve


```python
# ROC CURVE
fpr, tpr, thresholds = roc_curve(y_test.ravel(), rf_preds.ravel())
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```


![png](Random_Forest_files/Random_Forest_20_0.png)



```python

```
