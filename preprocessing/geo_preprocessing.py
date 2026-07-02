import os
import pandas as pd

# =====================================================
# PATHS
# =====================================================

INPUT_FILE = "dataset/geo_dataset/NFHS_5_India_Districts_Factsheet_Data.xls"
OUTPUT_FILE = "dataset/geo_dataset/NFHS_5_India_Districts_Factsheet_Data_Clean.csv"

print("=" * 50)
print("Geo Data Preprocessing Started...")
print("=" * 50)

# =====================================================
# LOAD EXCEL
# =====================================================

df = pd.read_excel(INPUT_FILE)

# =====================================================
# BASIC INFORMATION
# =====================================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Info:")
df.info()

# =====================================================
# MISSING VALUES
# =====================================================

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# DUPLICATES
# =====================================================

duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()

# =====================================================
# CLEAN COLUMN NAMES
# =====================================================

df.columns = df.columns.str.strip()

# =====================================================
# SAVE CLEAN DATA
# =====================================================

os.makedirs("dataset/geo_dataset", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nClean Geo Dataset Saved Successfully")

print("\nFinal Shape:")
print(df.shape)

print("=" * 50)
print("Geo Data Preprocessing Completed")
print("=" * 50)