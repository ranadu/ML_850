#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 00:50:30 2023

@author: robertanadu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import joblib




# Step 1
data = pd.read_csv('project 1 Data.csv')
print(data.head())






# Step 2

data = pd.read_csv('project 1 Data.csv')

steps = data['Step'].unique()

for step_label in steps:
    step_data = data[data['Step'] == step_label]
    
    step_mean = step_data.mean()
    step_std = step_data.std()
    
    print(f"Step {step_label} - Statistics")
    print(f"Mean:\n{step_mean}")
    print(f"Standard Deviation:\n{step_std}")

    for column in data.columns:
        if column != 'step':
            plt.figure(figsize=(6, 4))
            plt.hist(step_data[column], bins=20, alpha=0.5, label=column)
            plt.title(f'Step {step_label} - {column} Histogram')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()






#Step 3

data = pd.read_csv('project 1 Data.csv')
correlation_matrix = data.corr()
target_variable = 'Step'

correlations = correlation_matrix[target_variable]

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, fmt=".2f")
plt.title(f'Correlation Plot (Pearson) - {target_variable}')
plt.show()






# Step 4

data = pd.read_csv('project 1 Data.csv') 

X = data.drop('Step', axis=1) 
y = data['Step']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = {
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

param_grids = {
    'Decision Tree': {'max_depth': [None, 10, 20, 30]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.01, 0.001]}
}


best_models = {}
for clf_name, clf in classifiers.items():
    grid_search = GridSearchCV(clf, param_grids[clf_name], cv=5)
    grid_search.fit(X_train, y_train)
    best_models[clf_name] = grid_search.best_estimator_

# Evaluate the models on the test data
results = {}
for clf_name, clf in best_models.items():
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[clf_name] = accuracy

# Print the results
for clf_name, accuracy in results.items():
    print(f'{clf_name}: Accuracy - {accuracy:.2f}')






# Step 5

data = pd.read_csv('project 1 Data.csv')

X = data.drop('Step', axis=1)
y = data['Step']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()






# Step 6

data = pd.read_csv('project 1 Data.csv') 


X = data.drop('Step', axis=1)  
y = data['Step']

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
model.fit(X, y)


joblib.dump(model, 'maintenance_model.joblib')


loaded_model = joblib.load('maintenance_model.joblib')


coordinates = [
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
]

predictions = loaded_model.predict(coordinates)


for i, coord in enumerate(coordinates):
    print(f'Coordinates: {coord} => Predicted Step: {predictions[i]}')





