import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score 
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv("train.csv")
print(df.head())

print('*'*50)

print (df.info())

print('*'*50)

print(df.isnull().sum())

print('*'*50)

print(df.describe())

print('*'*50)

df.drop_duplicates()


data = df[['Age','Sex','Pclass','SibSp','Parch','Embarked','Fare']].copy()

data['Fare'] = np.log1p(data['Fare'])

data['Sex']= data['Sex'].map({'female':0 , 'male':1})

data['Embarked'] = data['Embarked'].fillna('S')
data['Embarked']= data['Embarked'].map({'S':0 , 'C':1 , 'Q':2})


data['Age'] = data['Age'].fillna(data['Age'].median())

print (data.head())
print('*'*50)
print(data.isnull().sum())

x= data[['Age','Sex','Pclass','SibSp','Parch','Embarked']]
y= data['Fare']

x_train ,x_test , y_train, y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

model= LinearRegression()
model.fit(x_train , y_train)

y_pred = model.predict(x_test)

print("R2 score" ,r2_score(y_test,y_pred))
print("mean square erroe ", mean_squared_error(y_test, y_pred))


plt.figure(figsize=(12,4))

plt.scatter(y_test, y_pred, alpha=0.5, c='b', label='predictions')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')

plt.xlabel('Actual Fare (log scale)')
plt.ylabel('prediction fare (log scale)')
plt.title('Actual vs Predicted Fare')

plt.legend()
plt.show()
