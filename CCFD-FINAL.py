# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 17:38:44 2018

@author: yukes
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv('creditcard.csv')
df= df.sample(frac=0.1, random_state=0)
x= df.iloc[:,:-1].values
y= df.iloc[:,30].values
df= df.sample(frac=0.1, random_state=0)

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
sns.countplot(x='Class', data=df)

print(x.shape)
print(y.shape)

#splitting the data into training testing 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3, random_state=0)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,accuracy_score
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

#predicting the test set
y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,y_pred)



#printing the score of predicted value
print(recall_score(y_test,y_pred,average=None))
print(accuracy_score(y_test,y_pred))


no_frauds = len(df[df['Class'] == 1])
non_fraud_indices = df[df.Class == 0].index
random_indices = np.random.choice(non_fraud_indices,no_frauds, replace=False)
fraud_indices = df[df.Class == 1].index
under_sample_indices = np.concatenate([fraud_indices,random_indices])
under_sample = df.loc[under_sample_indices]

## Plot the distribution of data for undersampling## Plot 
#%matplotlib inline
sns.countplot(x='Class', data=under_sample)

X_under = under_sample.loc[:,under_sample.columns != 'Class']
y_under = under_sample.loc[:,under_sample.columns == 'Class']
X_under_train, X_under_test, y_under_train, y_under_test = train_test_split(X_under,y_under,test_size = 0.3, random_state = 0)

lr_under = LogisticRegression()
lr_under.fit(X_under_train,y_under_train)
y_under_pred = lr_under.predict(X_under_test)
print(recall_score(y_under_test,y_under_pred))
print(accuracy_score(y_under_test,y_under_pred))


## Recall for the full data## Recal 
y_pred_full = lr_under.predict(x_test)
print(recall_score(y_test,y_pred_full))
print(accuracy_score(y_test,y_pred_full))

lr_balanced = LogisticRegression(class_weight = 'balanced')
lr_balanced.fit(x_train,y_train)
y_balanced_pred = lr_balanced.predict(x_test)
print(recall_score(y_test,y_balanced_pred))
print(accuracy_score(y_test,y_balanced_pred))