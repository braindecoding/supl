import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = "resultsallmiyawaki.csv"
df = pd.read_csv(file_path, delimiter=';')

# Convert FID column to numeric
df["FID"] = df["FID"].str.replace(",", ".").astype(float)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("FID Analysis for Miyawaki Dataset", fontsize=18, fontweight='bold')

# --- 1. Box Plot: FID vs Z ---
sns.boxplot(data=df, x="Z", y="FID", palette="Set2", ax=axes[0, 0])
axes[0, 0].set_title("Box Plot: FID vs Z", fontsize=12, pad=10)
axes[0, 0].set_xlabel("Latent Dimension (Z)")
axes[0, 0].set_ylabel("Frechet Inception Distance (FID)")

# --- 2. Box Plot: FID vs IDM ---
sns.boxplot(data=df, x="IDM", y="FID", palette="pastel", ax=axes[0, 1])
axes[0, 1].set_title("Box Plot: FID vs IDM", fontsize=12, pad=10)
axes[0, 1].set_xlabel("Intermediate Dimension (IDM)")
axes[0, 1].set_ylabel("Frechet Inception Distance (FID)")

# --- 3. Box Plot: FID vs BATCH ---
sns.boxplot(data=df, x="BATCH", y="FID", palette="coolwarm", ax=axes[1, 0])
axes[1, 0].set_title("Box Plot: FID vs Batch Size", fontsize=12, pad=15)  # Kurangi jarak judul subplot bawah
axes[1, 0].set_xlabel("Batch Size")
axes[1, 0].set_ylabel("Frechet Inception Distance (FID)")

# --- 4. Box Plot: FID vs ITERASI ---
sns.boxplot(data=df, x="ITERASI", y="FID", palette="magma", ax=axes[1, 1])
axes[1, 1].set_title("Box Plot: FID vs Iteration Count", fontsize=12, pad=15)  # Samakan jarak judul subplot bawah
axes[1, 1].set_xlabel("Number of Iterations")
axes[1, 1].set_ylabel("Frechet Inception Distance (FID)")

# **PENYESUAIAN PENTING**
plt.subplots_adjust(hspace=0.8)  # Jarak lebih besar antara subplot atas dan bawah
plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)  # Rotasi label X subplot bawah kiri
plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=60)  # Rotasi label X subplot bawah kanan
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Sesuaikan agar tidak terpotong

# Tampilkan plot
plt.show()
