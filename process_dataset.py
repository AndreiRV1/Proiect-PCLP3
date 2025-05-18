import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("vehicles_trimmed.csv")
# print(df.head())
# print(df.describe())
df = df.drop("id",axis=1)
df = df.drop("Unnamed: 0",axis=1)
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
# plt.show()

df = df.drop("condition",axis=1)

df["cylinders"].value_counts().plot(kind="bar")
plt.title("Cylinders distribution")
plt.xlabel("Cylinders")
plt.ylabel("Number of samples")

plt.savefig('./img/cylinder.png',format='png')
# plt.show()

df = df.drop("cylinders",axis=1)



df["drive"].value_counts().plot(kind="bar")
plt.title("Drive distribution")
plt.xlabel("Drive")
plt.ylabel("Number of samples")

plt.savefig('./img/drive.png',format='png')
# plt.show()

# print(df["drive"].describe())

df["drive"] = df["drive"].fillna(df["drive"].mode()[0])
df = pd.get_dummies(df, columns=["drive"])

# print(df['state'].nunique())
# print(df['region'].nunique())

df = df.drop("region",axis=1)

df["state"].value_counts().plot(kind="bar")
plt.title("State")
plt.xlabel("State")
plt.ylabel("Number of samples")

plt.savefig('./img/state.png',format='png')
# plt.show()
# print(df["state"].describe())

df["state"] = df["state"].astype("category").cat.codes



# print(df["description"].info())
df = df.drop("description",axis=1)
df = df.drop("model",axis=1)

print(df["title_status"].info())
print(df["title_status"].nunique())

df["title_status"].value_counts().plot(kind="bar")
plt.title("Title_status")
plt.xlabel("Title_status")
plt.ylabel("Number of samples")

plt.savefig('./img/title.png',format='png')
# plt.show()

# print(df["title_status"].describe())
df["title_status"] = df["title_status"].fillna(df["title_status"].mode()[0])

df["title_status"] = df["title_status"].astype("category").cat.codes


# print(df["manufacturer"].describe())

df["manufacturer"].value_counts().plot(kind="bar")
plt.title("Manufacturer")
plt.xlabel("Manufacturer")
plt.ylabel("Number of samples")

plt.savefig('./img/manufacturer.png',format='png')

# print(df["model"].describe())

df["manufacturer"] = df["manufacturer"].fillna(df["manufacturer"].mode()[0])
df["manufacturer"] = df["manufacturer"].astype("category").cat.codes


# print(df["fuel"].describe())

df["fuel"].value_counts().plot(kind="bar")
plt.title("Fuel type")
plt.xlabel("Fuel type")
plt.ylabel("Number of samples")
plt.savefig('./img/fuel.png',format='png')

df["fuel"] = df["fuel"].fillna(df["fuel"].mode()[0])

df["fuel"] = df["fuel"].astype("category").cat.codes


# print(df["size"].describe())

df["size"].value_counts().plot(kind="bar")
plt.title("Size type")
plt.xlabel("Size type")
plt.ylabel("Number of samples")
plt.savefig('./img/size.png',format='png')

df["size"] = df["size"].fillna(df["size"].mode()[0])
df["size"] = df["size"].astype("category").cat.codes


# print(df["transmission"].describe())

df["transmission"].value_counts().plot(kind="bar")
plt.title("transmission type")
plt.xlabel("transmission type")
plt.ylabel("Number of samples")
plt.savefig('./img/transmission.png',format='png')

df["transmission"] = df["transmission"].fillna(df["transmission"].mode()[0])
df["transmission"] = df["transmission"].astype("category").cat.codes

# print(df["type"].describe())

df["type"].value_counts().plot(kind="bar")
plt.title("Type")
plt.xlabel("Type")
plt.ylabel("Number of samples")
plt.savefig('./img/type.png',format='png')

df["type"] = df["type"].fillna(df["type"].mode()[0])
df["type"] = df["type"].astype("category").cat.codes

# print(df["paint_color"].describe())

df["paint_color"].value_counts().plot(kind="bar")
plt.title("paint_color")
plt.xlabel("paint_color")
plt.ylabel("Number of samples")
plt.savefig('./img/paint_color.png',format='png')

df["paint_color"] = df["paint_color"].fillna(df["paint_color"].mode()[0])
df["paint_color"] = df["paint_color"].astype("category").cat.codes



# print(df["year"].head())

plt.figure()
df["year"].hist(bins=10)
plt.xlabel("Year")
plt.ylabel("Frequency")
plt.title("Histogram of Year")
plt.grid(False)

plt.savefig("./img/year_hist.png", format="png")
df['year'] = df['year'].fillna(df['year'].mean()) 

Q1 = df['year'].quantile(0.25)
Q3 = df['year'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['year'] > lower_bound) & (df['year'] < upper_bound)]

# print(df["odometer"].describe())

Q1 = df['odometer'].quantile(0.25)
Q3 = df['odometer'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['odometer'] > lower_bound) & (df['odometer'] < upper_bound)]

plt.figure()
df["odometer"].hist(bins=1000)
plt.xlabel("Kms")
plt.ylabel("Frequency")
plt.title("Histogram of odometer")
plt.grid(False)

plt.savefig("./img/odometer.png", format="png")
df['odometer'] = df['odometer'].fillna(df['odometer'].mean()) 


print(df.info())
print(df.head())
