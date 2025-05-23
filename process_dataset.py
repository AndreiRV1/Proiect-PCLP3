import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def plot_text_field(df,name):
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.2)
    df[name].value_counts().plot(kind="bar")  
    plt.title(name+" distribution")
    plt.xlabel(name+" type")
    plt.ylabel("Number of samples")  
    plt.savefig("./img/"+name+".png",format='png')
    # plt.show()
    plt.close()
    
def plot_numeric_field(df,name):
    plt.figure(figsize=(8, 6))
    df[name].hist(bins=10)
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.title("Histogram of "+name)
    plt.grid(False)
    plt.savefig("./img/"+name+".png", format="png")
    plt.close()
    
def relation_numeric(df,name):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='price', y=name, data=df)
    plt.title("Violin Plot of "+name+ "by price")
    plt.savefig("./img/corr"+name+".png", format="png")
    plt.close()


df = pd.read_csv("vehicles_trimmed.csv")
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
df = df.drop("description",axis=1)
# print(df.info())


plot_text_field(df,"condition")
df = df.drop("condition",axis=1)


plot_text_field(df,"cylinders")
df = df.drop("cylinders",axis=1)

# print(df["drive"].describe())
# print(df["drive"].info())

plot_text_field(df,"drive")

df["drive"] = df["drive"].fillna(df["drive"].mode()[0])
df = pd.get_dummies(df, columns=["drive"])

# print(df['state'].nunique())
# print(df['region'].nunique())

df = df.drop("region",axis=1)

plot_text_field(df,"state")
# print(df["state"].describe())
df["state"] = df["state"].astype("category").cat.codes

# print(df["title_status"].info())
# print(df["title_status"].nunique())
# print(df["title_status"].describe())

plot_text_field(df,"title_status")

df["title_status"] = df["title_status"].fillna(df["title_status"].mode()[0])
df["title_status"] = df["title_status"].astype("category").cat.codes


df = df.drop("model",axis=1)

# print(df["manufacturer"].describe())
plot_text_field(df,"manufacturer")

df["manufacturer"] = df["manufacturer"].fillna(df["manufacturer"].mode()[0])
df["manufacturer"] = df["manufacturer"].astype("category").cat.codes

# print(df["fuel"].describe())

plot_text_field(df,"fuel")

df["fuel"] = df["fuel"].fillna(df["fuel"].mode()[0])
df["fuel"] = df["fuel"].astype("category").cat.codes


# print(df["size"].describe())

plot_text_field(df,"size")

df["size"] = df["size"].fillna(df["size"].mode()[0])
df["size"] = df["size"].astype("category").cat.codes
df = df.drop("size",axis=1)

# print(df["transmission"].describe())

plot_text_field(df,"transmission")

df["transmission"] = df["transmission"].fillna(df["transmission"].mode()[0])
df["transmission"] = df["transmission"].astype("category").cat.codes

# print(df["type"].describe())

plot_text_field(df,"type")

df["type"] = df["type"].fillna(df["type"].mode()[0])
df["type"] = df["type"].astype("category").cat.codes

# print(df["paint_color"].describe())

plot_text_field(df,"paint_color")

df["paint_color"] = df["paint_color"].fillna(df["paint_color"].mode()[0])
df["paint_color"] = df["paint_color"].astype("category").cat.codes

print(df["year"].describe())

df['year'] = df['year'].fillna(df['year'].mean()) 

Q1 = df['year'].quantile(0.25)
Q3 = df['year'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['year'] > lower_bound) & (df['year'] < upper_bound)]
df["year"] = df["year"].astype(int)
plot_numeric_field(df,"year")

print(df["odometer"].describe())

df['odometer'] = df['odometer'].fillna(df['odometer'].mean()) 
Q1 = df['odometer'].quantile(0.25)
Q3 = df['odometer'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['odometer'] > lower_bound) & (df['odometer'] < upper_bound)]
plot_numeric_field(df,"odometer")




# print(df["price"].head())
# print(df["price"].info())

df= df.dropna(subset="price")
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['price'] > lower_bound) & (df['price'] < upper_bound)]
print(df["price"].describe())
plot_numeric_field(df,"price")

scaler = MinMaxScaler()
df[['price', 'odometer','year']] = scaler.fit_transform(df[['price', 'odometer','year']])

correlation_matrix = df.corr()

plt.figure(figsize=(8, 6))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.savefig("./img/corr.png", format="png")


# print(df.info())
# print(df.head())


relation_numeric(df,'year')
relation_numeric(df,'odometer')

df.to_csv("car_cleaned.csv")