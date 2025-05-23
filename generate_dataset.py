import pandas as pd

dataset = pd.read_csv("vehicles.csv")
trimmed_dataset = dataset.sample(frac=1)
trimmed_dataset.to_csv("vehicles_trimmed.csv")
print(trimmed_dataset.info())