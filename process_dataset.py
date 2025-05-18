import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("vehicles_trimmed.csv")
# print(df.head())
# print(df.describe())
df = df.drop("id",axis=1)
df = df.drop("url",axis=1)
df = df.drop("posting_date",axis=1)
df = df.drop("lat",axis=1)
df = df.drop("long",axis=1)
df = df.drop("image_url",axis=1)
df = df.drop("VIN",axis=1)
df = df.drop("region_url",axis=1)


df = df.drop("county",axis=1)

df["condition"].value_counts().plot(kind="bar")
plt.title("Condition distribution")
plt.xlabel("Condition type")
plt.ylabel("Number of samples")

plt.savefig('condition.png',format='png')
plt.show()

df = df.drop("condition",axis=1)

df["cylinders"].value_counts().plot(kind="bar")
plt.title("Cylinders distribution")
plt.xlabel("Cylinders")
plt.ylabel("Number of samples")

plt.savefig('./img/cylinder.png',format='png')
plt.show()

df = df.drop("cylinders",axis=1)



df["drive"].value_counts().plot(kind="bar")
plt.title("Drive distribution")
plt.xlabel("Drive")
plt.ylabel("Number of samples")

plt.savefig('./img/drive.png',format='png')
plt.show()

df["drive"] = df["drive"].fillna(df["drive"].mode()[0])
df = pd.get_dummies(df, columns=["drive"])



print(df.info())
print(df.head())
