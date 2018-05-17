---
layout: post
title: "Exploratory Data Analysis"
author: "Richard Penfil"
categories: Algorithm
tags: [Decision Tree, Machine Learning]
image: tree_head.jpg
---

#### Below is an example of a Decision Tree model. I used dataset (found here:https://www.kaggle.com/mlg-ulb/creditcardfraud/data), which contains credit card transactions over a two day period. The model created attempts to identify fraudulent transactions based on the provided (masked) attributes. Please see the Random Forest page or EDA for the data preparation details.


```python
from sklearn.tree import DecisionTreeRegressor

select = sklearn.feature_selection.SelectKBest(k=10)
clf = sklearn.tree.DecisionTreeClassifier()

steps = [('feature_selection', select),
        ('decision_tree', clf)]

pipeline = sklearn.pipeline.Pipeline(steps)

# fit your pipeline on training
pipeline.fit( X_train, y_train )
# predictions
dt_preds = pipeline.predict( X_test )
# test the predictions
report = sklearn.metrics.classification_report( y_test, dt_preds )
print(report)

from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X_test)
mean_absolute_error(y_test, predicted_home_prices)
```

                 precision    recall  f1-score   support
    
              0       1.00      1.00      1.00     93825
              1       0.74      0.75      0.74       162
    
    avg / total       1.00      1.00      1.00     93987
    
    




    0.00084054177705427345




```python
#GRID SEARCH: Depth
param_grid = {'max_depth': np.arange(3, 10)}

tree = GridSearchCV(DecisionTreeClassifier(), param_grid)

tree.fit(X_train, y_train)
tree_preds = tree.predict_proba(X_test)[:, 1]
tree_performance = roc_auc_score(y_test, tree_preds)

print ('DecisionTree: Area under the ROC curve = {}'.format(tree_performance))
```

    DecisionTree: Area under the ROC curve = 0.9072536867625242
    


```python
#CONFUSION MATRIX
tree_class = (tree_preds >=.5)
cnf_matrix = pd.crosstab([y_test], [tree_class], rownames=['actual'], colnames=['tree_preds'])

ax = sns.heatmap(cnf_matrix, annot=True, fmt="d", cmap="YlGnBu",cbar=False)
plt.ylabel('True label')
plt.xlabel('Predicted label')
```




    <matplotlib.text.Text at 0x1de89aa2898>




![png](Decision_tree_files/Decision_tree_3_1.png)



```python
# ROC CURVE

fpr, tpr, thresholds = roc_curve(y_test.ravel(), tree_preds.ravel())
dt_roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% dt_roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```


![png](Decision_tree_files/Decision_tree_4_0.png)



```python

```
