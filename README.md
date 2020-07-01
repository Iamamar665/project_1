# project_1
Wine Quality data set


# importing the data set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\Amar Shilvanth\Downloads\\datasets_4458_8204_winequality-red.csv")

data


# checking for null values

data.isnull().sum()

# univariate analysis
## Histograms

import seaborn as sns


sns.set(rc={'figure.figsize':(15,8)})
sns.distplot(data['citric acid'], bins=30, color = "blue")
plt.show()

sns.set(rc={'figure.figsize':(15,8)})
sns.distplot(data['residual sugar'], bins=30, color = "red")
plt.show()

sns.set(rc={'figure.figsize':(15,8)})
sns.distplot(data['alcohol'], bins=30, color = "green")
plt.show()

sns.set(rc={'figure.figsize':(15,8)})
sns.distplot(data['pH'], bins=30, color = "violet")
plt.show()

sns.set(rc={'figure.figsize':(15,8)})
sns.distplot(data['density'], bins=30, color = "indigo")
plt.show()

sns.set(rc={'figure.figsize':(15,8)})
sns.distplot(data['fixed acidity'], bins=30, color = "indigo")
plt.show()

## boxplots

boxplot_1 = sns.boxplot(y=data["fixed acidity"], palette = "Greens")

boxplot_2 = sns.boxplot(y=data["volatile acidity"], palette = "Reds")

boxplot_3 = sns.boxplot(y=data["citric acid"], palette = "Blues")

boxplot_4 = sns.boxplot(y=data["residual sugar"], palette = "Reds")

boxplot_5 = sns.boxplot(y=data["chlorides"], palette = "Greens")

boxplot_6 = sns.boxplot(y=data["free sulfur dioxide"], palette = "Greens")

boxplot_7 = sns.boxplot(y=data["total sulfur dioxide"], palette = "Greens")

boxplot_8 = sns.boxplot(y=data["density"], palette = "Greens")

boxplot_9 = sns.boxplot(y=data["pH"], palette = "Greens")

boxplot_10 = sns.boxplot(y=data["sulphates"], palette = "Greens")

boxplot_11 = sns.boxplot(y=data["alcohol"], palette = "Greens")

boxplot_12 = sns.boxplot(y=data["quality"], palette = "Greens")

# Bivariate Analysis
## scatter plots

x1 = data["fixed acidity"]
y1 = data["quality"]
plt.scatter(x1,y1, alpha = 0.5)
plt.xlabel("fixed acidity")
plt.ylabel("quality")

x2 = data["residual sugar"]
y2 = data["alcohol"]
plt.scatter(x2,y2,alpha = 0.5)
plt.xlabel("residual sugar")
plt.ylabel("quality")

x3 = data["citric acid"]
y3 = data["pH"]
plt.scatter(x3,y3,alpha = 0.5)
plt.xlabel("citric acid")
plt.ylabel("pH")

x4 = data["density"]
y4 = data["alcohol"]
plt.scatter(x4,y4,alpha = 0.5)
plt.xlabel("density")
plt.ylabel("alchohol")

x5 = data["pH"]
y5 = data["quality"]
plt.scatter(y5,x5,alpha = 0.5)
plt.xlabel("quality")
plt.ylabel("pH")

# Modelling the dataset

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

## splitting the data into target and label data
y = data.quality
X = data.drop('quality', axis=1) # axis=1 determines here column

# splitting the data into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

# preproccessing the data helps machine to understand the data easily. it's not necessary
X_train_scaled = preprocessing.scale(X_train)
print(X_train_scaled)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train) # fitting the data for training

## checking how efficiently the algorithm is predicting the label (in this case wine quality).
confidence = knn.score(X_test, y_test)
print("\nThe confidence score:\n")
print(confidence)

y_pred = knn.predict(X_test)  # checking the predicted label values


#converting the numpy array to list
x=np.array(y_pred).tolist()

#printing first 100 predictions
print("\nThe prediction:\n")
for i in range(0,100):
    print(x[i])
    
#printing first 100 expectations
print("\nThe expectation:\n")
print(y_test.head(100))
