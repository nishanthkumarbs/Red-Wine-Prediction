import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics

# Load dataset
df = pd.read_csv('winequality-red.csv')

# Check for null values
print("Null values in each column:")
print(df.isnull().sum())

# Dataset info
print("Dataset information:")
print(df.info())

# Dataset description
print("Dataset description:")
print(df.describe())

# Value counts of 'quality' column
print("Value counts of 'quality' column:")
print(df['quality'].value_counts())

# Convert 'quality' to binary classification
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
df.rename(columns={'quality': 'good-quality'}, inplace=True)
print(df.head())

# Plot count of good vs bad quality wines
plt.figure(figsize=(5, 5))
sns.countplot(x='good-quality', data=df)
plt.xlabel('Good Quality')
plt.ylabel('Count')
plt.title('Count of Good vs Bad Quality Wines')
plt.show()

# Correlation plot
df.corr()['good-quality'][:-1].sort_values().plot(kind='bar')
plt.show()

# Heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True)
plt.show()

# Scatter plots for feature relationships
fig, ax = plt.subplots(2, 4, figsize=(20, 20))
sns.scatterplot(x='fixed acidity', y='citric acid', hue='good-quality', data=df, ax=ax[0, 0])
sns.scatterplot(x='volatile acidity', y='citric acid', hue='good-quality', data=df, ax=ax[0, 1])
sns.scatterplot(x='free sulfur dioxide', y='total sulfur dioxide', hue='good-quality', data=df, ax=ax[0, 2])
sns.scatterplot(x='fixed acidity', y='density', hue='good-quality', data=df, ax=ax[0, 3])
sns.scatterplot(x='fixed acidity', y='pH', hue='good-quality', data=df, ax=ax[1, 0])
sns.scatterplot(x='citric acid', y='pH', hue='good-quality', data=df, ax=ax[1, 1])
sns.scatterplot(x='chlorides', y='sulphates', hue='good-quality', data=df, ax=ax[1, 2])
sns.scatterplot(x='residual sugar', y='alcohol', hue='good-quality', data=df, ax=ax[1, 3])
plt.show()

# Data preparation for modeling
X = df.drop('good-quality', axis=1)
y = df['good-quality']
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Logistic Regression Accuracy: ", accuracy_score(y_test, lr_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, cmap='Blues')
plt.title("Confusion Matrix for Logistic Regression")
plt.show()

# Support Vector Machine model
sv = svm.SVC()
sv.fit(X_train, y_train)
sv_pred = sv.predict(X_test)
print("Support Vector Machine Accuracy: ", accuracy_score(y_test, sv_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, sv_pred))
sns.heatmap(confusion_matrix(y_test, sv_pred), annot=True, cmap='Reds')
plt.title("Confusion Matrix for Support Vector Machine")
plt.show()

# Decision Tree model
tr = DecisionTreeClassifier()
tr.fit(X_train, y_train)
tr_pred = tr.predict(X_test)
print("Decision Tree Accuracy: ", accuracy_score(y_test, tr_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, tr_pred))
sns.heatmap(confusion_matrix(y_test, tr_pred), annot=True, cmap='Greens')
plt.title("Confusion Matrix for Decision Tree")
plt.show()

# K-Nearest Neighbors model
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(X_train, y_train)
kn_pred = kn.predict(X_test)
print("K-Nearest Neighbors Accuracy: ", accuracy_score(y_test, kn_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, kn_pred))
sns.heatmap(confusion_matrix(y_test, kn_pred), annot=True, cmap='Purples')
plt.title("Confusion Matrix for K-Nearest Neighbors")
plt.show()

# Model accuracy comparison
models = ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'K-Nearest Neighbors']
accuracy = [accuracy_score(y_test, lr_pred), accuracy_score(y_test, sv_pred), accuracy_score(y_test, tr_pred), accuracy_score(y_test, kn_pred)]
plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=accuracy)
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.7, 1.0)
plt.show()