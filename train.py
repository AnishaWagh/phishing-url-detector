import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import DOMAIN_FEATURE_NAMES

df = pd.read_csv("data/features.csv")
X = df[DOMAIN_FEATURE_NAMES]
y = df["is_phishing"]  # 1 = phishing, 0 = legitimate

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        n_jobs=-1, random_state=42, eval_metric="logloss",
    ),
}

best_name, best_model, best_f1 = None, None, -1

for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print(classification_report(y_test, preds, target_names=["legitimate", "phishing"]))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_test, preds))

    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_name, best_model, best_f1 = name, model, f1

print(f"\nBest model: {best_name} (phishing F1 = {best_f1:.4f})")

importances = pd.Series(best_model.feature_importances_, index=DOMAIN_FEATURE_NAMES)
print("\nTop 10 features:\n", importances.sort_values(ascending=False).head(10).round(4))

joblib.dump(
    {"model": best_model, "features": DOMAIN_FEATURE_NAMES},
    "models/phishing_model.pkl",
    compress=3,
)
print("\nSaved models/phishing_model.pkl")