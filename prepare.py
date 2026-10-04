import pandas as pd
from features import extract_features

df = pd.read_csv("data/phishing.csv")[["URL", "label"]]

# Drop duplicates, then flip the label so 1 = phishing
df = df.drop_duplicates(subset="URL").reset_index(drop=True)
df["is_phishing"] = 1 - df["label"]

print("Rows after cleaning:", len(df))
print(df["is_phishing"].value_counts())

print("Extracting features (this can take a minute or two)...")
feats = pd.DataFrame([extract_features(u) for u in df["URL"]])
out = pd.concat([df[["URL", "is_phishing"]], feats], axis=1)

out.to_csv("data/features.csv", index=False)
print("Saved data/features.csv with shape", out.shape)
print(out.groupby("is_phishing").mean(numeric_only=True).T.round(3))