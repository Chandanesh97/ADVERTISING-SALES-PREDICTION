import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
df = pd.read_csv("dataset/cleaned_advertising_data.csv")

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Histograms for feature distributions
print("\nGenerating Histograms...")
df.hist(figsize=(10, 6), bins=20)
plt.suptitle("Feature Distributions")
plt.show()

# Scatter plots: Each feature vs. Sales
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# TV vs Sales
print("\nGenerating Scatter Plots...")
axes[0].scatter(df["TV"], df["Sales"], alpha=0.6)
axes[0].set_xlabel("TV Advertising Budget")
axes[0].set_ylabel("Sales")
axes[0].set_title("TV vs Sales")

# Radio vs Sales
axes[1].scatter(df["Radio"], df["Sales"], alpha=0.6, color="red")
axes[1].set_xlabel("Radio Advertising Budget")
axes[1].set_ylabel("Sales")
axes[1].set_title("Radio vs Sales")

# Newspaper vs Sales
axes[2].scatter(df["Newspaper"], df["Sales"], alpha=0.6, color="green")
axes[2].set_xlabel("Newspaper Advertising Budget")
axes[2].set_ylabel("Sales")
axes[2].set_title("Newspaper vs Sales")

plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()