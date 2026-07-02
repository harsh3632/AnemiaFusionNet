import os
import pandas as pd

# =====================================================
# PATHS
# =====================================================

INPUT_FILE = "dataset/clinical_dataset/IDA_dataset.csv"
OUTPUT_FILE = "dataset/clinical_dataset/IDA_dataset_clean.csv"

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 50)
print("Clinical Data Preprocessing Started...")
print("=" * 50)

df = pd.read_csv(INPUT_FILE)

# =====================================================
# BASIC INFORMATION
# =====================================================

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())

# =====================================================
# MISSING VALUES
# =====================================================

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# DUPLICATE VALUES
# =====================================================

duplicates = df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicate rows removed.")

# =====================================================
# CLEAN COLUMN NAMES
# =====================================================

df.columns = df.columns.str.strip()

# =====================================================
# SAVE CLEAN DATASET
# =====================================================

os.makedirs("dataset/clinical_dataset", exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nClean dataset saved successfully.")

print("\nFinal Dataset Shape:")
print(df.shape)

print("=" * 50)
print("Clinical Data Preprocessing Completed")
print("=" * 50)