import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

corr = df.corr()

print(corr["stress_level"].sort_values(ascending=False))

print("\nMenampilkan plot Heatmap...")

# Membuat ukuran kanvas grafik
plt.figure(figsize=(18, 12))

# Menggambar heatmap dari nilai korelasi
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

# Memberi judul dan menampilkan grafik
plt.title("Heatmap Korelasi", fontsize=16)
plt.show()

# --- STEP 2: FEATURE SELECTION (EKSEKUSI SNIPER) ---
print("\nMelakukan Feature Selection...")

# Menyingkirkan 3 fitur yang korelasinya di bawah absolut 0.6
kolom_dibuang = ["blood_pressure", "breathing_problem", "living_conditions"]
df_clean = df.drop(columns=kolom_dibuang)

print("Fitur yang resmi dibuang :", kolom_dibuang)
print("Dimensi dataset awal     :", df.shape)
print("Dimensi dataset bersih   :", df_clean.shape)