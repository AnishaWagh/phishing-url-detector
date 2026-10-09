"""
explore.py
Quick look at the raw dataset: size, columns, label counts, duplicates and sample URLs.
"""
import pandas as pd

df = pd.read_csv("data/phishing.csv")            # load the raw dataset

print("Shape:", df.shape)                        # (number of rows, number of columns)
print("\nColumns:", list(df.columns))            # names of all columns
print("\nFirst 5 rows:\n", df.head())            # preview of the first rows
print("\nColumn types:\n", df.dtypes)            # data type of each column

# Find the URL and label columns automatically by checking likely names.
url_col = next((c for c in df.columns if c.lower() in ("url", "urls")), None)
label_col = next((c for c in df.columns if c.lower() in ("label", "type", "class", "status", "result")), None)
print("\nDetected URL column:", url_col)
print("Detected label column:", label_col)

if label_col:
    print("\nLabel counts:\n", df[label_col].value_counts())   # how many rows per class
if url_col:
    print("\nMissing URLs:", df[url_col].isna().sum())         # empty URL cells
    print("Duplicate URLs:", df[url_col].duplicated().sum())   # repeated URLs
    print("\nSample URLs:\n", df[url_col].sample(5, random_state=1).to_string())  # 5 random URLs

# Print 10 real phishing URLs (label 0 in this dataset). Paste as TEXT only; never open them.
print(df[df["label"] == 0]["URL"].sample(10, random_state=3).to_string())