import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from features import extract_domain_features

bundle = joblib.load("models/phishing_model.pkl")
model, names = bundle["model"], bundle["features"]

df = pd.read_csv("data/features.csv")
X, y = df[names], df["is_phishing"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
preds = model.predict(X_test)

print(classification_report(y_test, preds, target_names=["legitimate", "phishing"], digits=4))
print(confusion_matrix(y_test, preds))
print("\nTop 10 features:\n", pd.Series(model.feature_importances_, index=names)
      .sort_values(ascending=False).head(10).round(4))

tests = {
    "LEGIT  google home":     "https://www.google.com",
    "LEGIT  google search":   "https://www.google.com/search?q=python+tutorial&hl=en",
    "LEGIT  github repo":     "https://github.com/microsoft/vscode",
    "LEGIT  wikipedia page":  "https://en.wikipedia.org/wiki/Machine_learning",
    "LEGIT  amazon product":  "https://www.amazon.in/dp/B0C1234567?ref=nav_cart",
    "LEGIT  irctc":           "https://irctc.co.in",
    "LEGIT  http only":       "http://example.com",
    "PHISH  fake paypal":     "http://secure-login-paypal-verify.com/account/update",
    "PHISH  ip address":      "http://192.168.1.5/login.php",
    "PHISH  crypto bonus":    "http://free-crypto-bonus-claim.top/wallet/confirm",
    "PHISH  subdomain trick": "http://paypal.com.account-verify.example-secure.xyz/signin",
    "PHISH  @ trick":         "http://bank-update-password@malicious-site.ru/login",
}
print("\nReal-world checks (phishing probability):")
for label, u in tests.items():
    p = model.predict_proba(pd.DataFrame([extract_domain_features(u)])[names])[0][1]
    print(f"{p:6.1%}  {label:24s} {u}")