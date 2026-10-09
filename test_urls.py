"""
test_urls.py
Runs the saved model on a short list of URLs and reports how many it got right.
"""
import joblib
import pandas as pd
from features import extract_domain_features

bundle = joblib.load("models/phishing_model.pkl")   # load model + feature names
model, names = bundle["model"], bundle["features"]

# (expected answer, URL). Add your own lines here to test more links.
urls = [
    ("LEGIT", "https://www.google.com"),
    ("LEGIT", "https://www.irctc.co.in"),
    ("LEGIT", "https://github.com/microsoft/vscode"),
    ("LEGIT", "https://www.sbi.co.in"),
    ("PHISH", "http://secure-login-paypal-verify.com/account/update"),
    ("PHISH", "http://amaz0n-india-support-verify.com"),
    ("PHISH", "http://sbi-netbanking-login-update.xyz"),
    ("PHISH", "http://192.168.1.5/login.php"),
]

correct = 0                                          # counter of right answers
for expected, u in urls:
    # Probability (0 to 1) that this URL is phishing.
    p = model.predict_proba(pd.DataFrame([extract_domain_features(u)])[names])[0][1]
    got = "PHISH" if p >= 0.5 else "LEGIT"           # 0.5 is the decision threshold
    correct += got == expected                       # adds 1 when the answer matches
    print(f"{'OK ' if got == expected else 'BAD'} {p:6.1%} expected {expected} got {got}  {u}")
print(f"\n{correct}/{len(urls)} correct")