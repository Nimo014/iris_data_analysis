#import module
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
random.seed(1)
#read data
df = pd.read_csv('iris.csv')


df = df.drop(columns='Id')

corr = df.corr()

fig , ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr ,annot=True,cmap='coolwarm', ax = ax)
plt.show()


#preprocessing
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

#training data
x = df.loc[:,df.columns[:-1]]
y = df['Species']

X_train,X_test,Y_train,y_test = train_test_split(x,y,test_size = 0.3)

lr = LogisticRegression()
lr.fit(X_train,Y_train)

KNC = KNeighborsClassifier()
KNC.fit(X_train,Y_train)

clf = DecisionTreeClassifier()
clf.fit(X_train,Y_train)


print((lr.score(X_test,y_test)*100))
print(KNC.score(X_test,y_test)*100)
print(clf.score(X_test,y_test)*100)
