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

print(df.info())
print(df.head())
print(df["cylinders"].info())
print(df["condition"].info())