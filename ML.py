import pandas as pd

# DF = Data Frame
df = pd.read_csv("StressLevelDataset.csv")

print(df.shape)
print(df.info())
print(df.head())
print(df.describe())

print("Jumlah mahasiswa:")
print(df.shape[0])

print("\nRata-rata anxiety:")
print(round(df["anxiety_level"].mean(),2))

print("\nRiwayat masalah mental:")
print(df["mental_health_history"].sum())

print(df["stress_level"].value_counts())

print()

print(df["stress_level"].value_counts(normalize=True)*100)