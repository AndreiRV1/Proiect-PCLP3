from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.svm import SVC
import joblib

df = pd.read_csv("car_cleaned.csv")

X = df.drop(columns="price",axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 20) 
print(1)
model = RandomForestRegressor()
model.fit(X_train,y_train)
print(2)

y_pred = model.predict(X_test)
print(3)

accuracy = root_mean_squared_error(y_test, y_pred)
print(f"Acuratețea modelului de regresie logistică: {accuracy:.2f}")

joblib.dump(model,"model.pk1")