import pandas as pd

# Loading the advertising sales dataset
df = pd.read_csv("dataset/advertising_dataset.csv")

# Displaying the basic information
print("\nBasic Information:")
print(df.info())

# Checking for nay missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Droping the unnecessary columns (like S.no,because it's just a serial number)
if "S.no" in df.columns:
    df = df.drop(columns=["S.no"])

# Checking for duplicate values in the dataset
duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

# Save the cleaned dataset
df.to_csv("dataset/cleaned_advertising_data.csv", index=False)
print("\nData Preprocessing Completed! Cleaned data saved as 'cleaned_advertising_data.csv'.")