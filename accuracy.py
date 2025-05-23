import joblib
from sklearn.metrics import root_mean_squared_error, mean_absolute_error,r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("car_cleaned.csv")

X = df.drop(columns="price",axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 20) 

model = joblib.load("model.pk1")

y_pred = model.predict(X_test)
print(3)

accuracy_rms = root_mean_squared_error(y_test, y_pred)
accuracy_mae = mean_absolute_error(y_test,y_pred)
accuracy_r2 = r2_score(y_test,y_pred)

print(f"Acuratețea modelului de regresie logistică(rms): {accuracy_rms:.2f}")
print(f"Acuratețea modelului de regresie logistică(mae): {accuracy_mae:.2f}")
print(f"Acuratețea modelului de regresie logistică(r2): {accuracy_r2:.2f}")

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (True - Predicted)')
plt.title('Residual Plot')
plt.savefig("./img/residuals.png",format='png')



