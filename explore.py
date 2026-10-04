import pandas as pd

df = pd.read_csv("data/phishing.csv")

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst 5 rows:\n", df.head())
print("\nColumn types:\n", df.dtypes)

url_col = next((c for c in df.columns if c.lower() in ("url", "urls")), None)
label_col = next((c for c in df.columns if c.lower() in ("label", "type", "class", "status", "result")), None)
print("\nDetected URL column:", url_col)
print("Detected label column:", label_col)

if label_col:
    print("\nLabel counts:\n", df[label_col].value_counts())
if url_col:
    print("\nMissing URLs:", df[url_col].isna().sum())
    print("Duplicate URLs:", df[url_col].duplicated().sum())
    print("\nSample URLs:\n", df[url_col].sample(5, random_state=1).to_string())