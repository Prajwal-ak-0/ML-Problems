import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
dataset = pd.read_csv('onlinefoods.csv')
X = dataset.iloc[:,[0,1,2,3,4,5,6,11]]
Y = dataset.iloc[:,10]

# Store the column names before transforming X
column_names = X.columns

# Apply OneHotEncoder to all columns of X
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), slice(None))], remainder='passthrough')
X = np.array(ct.fit_transform(X).toarray())

# Apply LabelEncoder to Y
le = LabelEncoder()
Y = le.fit_transform(Y)

# Split the dataset into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 1)

# Apply StandardScaler to the first and seventh columns
sc = StandardScaler()
X_train[:, 0] = sc.fit_transform(X_train[:, 0].reshape(-1, 1)).ravel()
X_test[:, 0] = sc.transform(X_test[:, 0].reshape(-1, 1)).ravel()
X_train[:, 6] = sc.fit_transform(X_train[:, 6].reshape(-1, 1)).ravel()
X_test[:, 6] = sc.transform(X_test[:, 6].reshape(-1, 1)).ravel()

# Train the Logistic Regression model
classifier = LogisticRegression(random_state = 0, max_iter=1000)
classifier.fit(X_train, Y_train)

# Predict the test set results
Y_pred = classifier.predict(X_test)

print(Y_pred)
# Print the confusion matrix and accuracy score
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
accuracy = accuracy_score(Y_test, Y_pred)
print(accuracy)

