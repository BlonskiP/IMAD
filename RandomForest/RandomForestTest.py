#Script to test random forest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

iris = pd.read_csv('datasets\Iris.csv')
#iris = preprocess_dateset(iris,['Species'],['Id'])

X = iris.iloc[:,1:-1] #Iris with no ID and Species columns
Y = iris.iloc[:,-1] # Iris Species columns (last one)

x_train , x_test, y_train,  y_test = train_test_split(X,Y,test_size=0.2)
print("Single decision tree")
decisionTree = DecisionTreeClassifier()
decisionTree.fit(x_train,y_train)
print(classification_report(y_test, decisionTree.predict(x_test), output_dict=False,zero_division=1))

print("Random Forest - 10 trees")
randomForest = RandomForestClassifier(n_estimators=100, # how many decision trees inside?
                                      n_jobs = -1, #use all processors for pararell operations

                                      )
randomForest.fit(x_train,y_train)

print(classification_report(y_test, randomForest.predict(x_test), output_dict=False,zero_division=1))

