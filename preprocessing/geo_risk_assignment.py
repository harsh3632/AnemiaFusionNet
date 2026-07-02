import os
import pandas as pd

# =====================================================
# PATHS
# =====================================================

INPUT_FILE = "dataset/geo_dataset/NFHS_5_India_Districts_Factsheet_Data_Clean.csv"
OUTPUT_FILE = "dataset/geo_dataset/geo_risk_data.csv"

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("Geo Risk Assignment Started...")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

# =====================================================
# REQUIRED COLUMNS
# =====================================================

district_col = "District Names"
state_col = "State/UT"
anemia_col = "All women age 15-49 years who are anaemic22 (%)"

# Keep only required columns
geo_df = df[[district_col, state_col, anemia_col]].copy()

# Rename columns
geo_df.columns = [
    "District",
    "State",
    "Anemia_Prevalence"
]

# =====================================================
# ASSIGN RISK
# =====================================================

def assign_risk(value):
    if value < 40:
        return "Low"
    elif value < 50:
        return "Medium"
    else:
        return "High"

geo_df["Geo_Risk"] = geo_df["Anemia_Prevalence"].apply(assign_risk)

# =====================================================
# SAVE
# =====================================================

os.makedirs("dataset/geo_dataset", exist_ok=True)

geo_df.to_csv(OUTPUT_FILE, index=False)

# =====================================================
# SUMMARY
# =====================================================

print("\nGeo Risk Distribution")
print(geo_df["Geo_Risk"].value_counts())

print("\nOutput Shape:", geo_df.shape)

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 60)
print("Geo Risk Assignment Completed")
print("=" * 60)