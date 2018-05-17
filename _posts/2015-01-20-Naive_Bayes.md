---
layout: post
title: "Naive Bayes"
author: "Richard Penfil"
categories: Algorithm
tags: [Naive Bayes, Machine Learning]
image: bayes.png
---

#### Below is an example of a Naive Bayes model. I used dataset (found here:https://www.kaggle.com/mlg-ulb/creditcardfraud/data), which contains credit card transactions over a two day period. The model created attempts to identify fraudulent transactions based on the provided (masked) attributes. Please see the Random Forest page or EDA for the data preparation details.


```python
gnb = BernoulliNB()
nb_preds = gnb.fit(X_train, y_train).predict(X_test)
mean_absolute_error(y_test, nb_preds)

NB_performance = roc_auc_score(y_test, nb_preds)
print ('DecisionTree: Area under the ROC curve = {}'.format(NB_performance))
```

    DecisionTree: Area under the ROC curve = 0.8208544275690559
    


```python
#Confusion Matrix
cnf_matrix = confusion_matrix(y_test,nb_preds)
np.set_printoptions(precision=2)

ax = sns.heatmap(cnf_matrix, annot=True, fmt="d", cmap="YlGnBu",cbar=False)
plt.ylabel('True label')
plt.xlabel('Predicted label')
```




    <matplotlib.text.Text at 0x1de89aeed30>




![png](Naive_Bayes_files/Naive_Bayes_2_1.png)



```python
# ROC CURVE

fpr, tpr, thresholds = roc_curve(y_test.ravel(), nb_preds.ravel())
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


![png](Naive_Bayes_files/Naive_Bayes_3_0.png)



```python
#NB
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_preds)
nb_auc = auc(fpr_nb,tpr_nb)
#DT
fpr_dt, tpr_dt, _ = roc_curve(y_test, tree_preds )
dt_auc = auc(fpr_dt,tpr_dt)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_nb, tpr_nb, label='Naive Bayes AUC = %0.2f'% nb_auc)
plt.plot(fpr_dt, tpr_dt, label='Decision Tree AUC = %0.2f'% dt_auc)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
```


![png](Naive_Bayes_files/Naive_Bayes_4_0.png)


Based on the receiver operating characteristic curve, the decision tree model is superior to naive bayes on this dataset, as the area under the curve is larger. 


```python

```
