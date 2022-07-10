# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Reading dataset
iris = pd.read_csv("Iris.csv")
print(iris.columns)
print(iris.head())

# Define target attribute (species)
y = iris['species']
iris.drop(columns='species',inplace=True)
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

# Training the model
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)

# Create pickle file
pickle.dump(model,open('model.pkl','wb'))
