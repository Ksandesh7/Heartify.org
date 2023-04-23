# %%
import pandas as pd #To read CSV file
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #To split data into trainning and testing
from sklearn.preprocessing import StandardScaler # To scale numerical values from -1 to 1 range

from sklearn.linear_model import LogisticRegression # import Logistic regression
from sklearn.tree import DecisionTreeClassifier # import Decision tree
from sklearn.ensemble import RandomForestClassifier # import Random Forest Classifier
from sklearn.svm import SVC # import SVM
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score # accuracy used for accuracy of model
from sklearn.metrics import confusion_matrix

# %%
df = pd.read_csv("HeartAttack.csv",na_values = '?')

# %%
df.head(10) #Top 10 row


# %%
df = df.drop(columns = ["slope","ca","thal"],axis = 1)

# %%
df.head(10) #Top 10 row

# %%
df = df.dropna()

# %%
df.head()

# %%
numerical_cols = ["age","trestbps","chol","thalach","oldpeak"]  # We have to standardized this numerical columns
cat_cols = list(set(df.columns)-set(numerical_cols)-{"target"})

# %%
numerical_cols

# %%
cat_cols

# %%
df_train , df_test = train_test_split(df,test_size = 0.2, random_state = 42)

# %%
len(df_train) , len(df_test)

# %%
scaler = StandardScaler()  # object of Scaler

# %%
def get_features_and_target_arrays(df,numerical_cols,cat_cols,scaler):
  x_numeric_scaled = scaler.fit_transform(df[numerical_cols])  # Numeric Columns scaled and converted to numpy array
  #x_categorical = df[cat_cols].tonumpy() # categorical Columns converted to numpy array
  x_categorical = df[cat_cols].values  # same as above statement
  x = np.hstack((x_categorical,x_numeric_scaled))  # Merge x_categorical and x_numeric to x
  y = df["target"]  # on y there will be target
  return x,y

# %%
x_train,y_train = get_features_and_target_arrays(df_train,numerical_cols,cat_cols,scaler)

# %%
x_train  # Basically gives x_train array and vice versa for y_train by using "y_train"


# %%
### Train using Logistic Regression
clf = LogisticRegression() # Object of logistic regression
clf.fit(x_train,y_train)  # fit method will map x_train with corresponding y_train

# %%
x_test,y_test = get_features_and_target_arrays(df_test,numerical_cols,cat_cols,scaler) 


# %%
test_pred = clf.predict(x_test)


# %%
mean_squared_error(y_test,test_pred)


# %%
accuracy_score(y_test,test_pred)  # Accuracy of Logistic Regression


# %%
confusion_matrix(y_test,test_pred) # Confusion matrix of result


# %%



