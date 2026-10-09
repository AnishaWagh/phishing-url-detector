"""
prepare.py
Reads the raw dataset, cleans it, turns every URL into features and saves
the result to data/features.csv. Run this BEFORE train.py.
"""
import pandas as pd                             # tables (DataFrames) and CSV reading
from features import extract_domain_features    # our function that makes features from a URL

# Load the raw dataset and keep only the URL column and the label column.
df = pd.read_csv("data/phishing.csv")[["URL", "label"]]

# Remove repeated URLs, then renumber the rows from 0.
df = df.drop_duplicates(subset="URL").reset_index(drop=True)
# In this dataset 1 = legitimate, 0 = phishing. Flip it so 1 = phishing (easier to understand).
df["is_phishing"] = 1 - df["label"]

print("Rows after cleaning:", len(df))           # how many URLs remain
print(df["is_phishing"].value_counts())          # how many phishing (1) and legitimate (0)

print("Extracting domain features (this can take a minute or two)...")
# Run the feature function on every URL and put all results in one table.
feats = pd.DataFrame([extract_domain_features(u) for u in df["URL"]])
# Join URL + label with the new feature columns side by side.
out = pd.concat([df[["URL", "is_phishing"]], feats], axis=1)

out.to_csv("data/features.csv", index=False)     # save the table for train.py
print("Saved data/features.csv with shape", out.shape)
# Average of each feature for legitimate (0) vs phishing (1): shows which features differ.
print(out.groupby("is_phishing").mean(numeric_only=True).T.round(3))