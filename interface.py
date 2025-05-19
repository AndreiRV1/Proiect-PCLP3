import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,root_mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("car_cleaned.csv")

X = df.drop(columns="price",axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 20) 

model = joblib.load("model2.pk1")

y_pred = model.predict(X_test)
print(3)

accuracy = root_mean_squared_error(y_test, y_pred)

print(f"Acuratețea modelului de regresie logistică: {accuracy:.2f}")


