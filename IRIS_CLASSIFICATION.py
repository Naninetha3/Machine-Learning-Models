import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

iris = load_iris()
datafrm = pd.DataFrame(data=iris.data, columns=iris.feature_names)
datafrm['species'] = [iris.target_names[i] for i in iris.target]  # Add species column manually

# Print DataFrame
print(datafrm.head())

X = iris.data
y=iris.target
print("feature names:",iris.feature_names)
print("target names:",iris.target_names)
print(X)
print(y)


sns.pairplot(iris, hue='species')
print(iris.columns)
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

model = LogisticRegression()
model.fit(X,y)

result = model.predict([ [6.5, 3.,  5.2, 2. ]
])
predicted_name = iris.target_names[result][0]
X_train,X_test , y_train ,y_test=train_test_split(X,y,test_size=0.2,random_state=45)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy of the model:",accuracy)
print(predicted_name)