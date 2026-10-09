"""
train.py
Trains two models (Random Forest and XGBoost), compares them on unseen data,
and saves the better one to models/phishing_model.pkl.
"""
import joblib                                    # saves/loads Python objects (our model) to a file
import pandas as pd                              # tables
from sklearn.ensemble import RandomForestClassifier   # model 1: many decision trees voting
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # scoring tools
from sklearn.model_selection import train_test_split  # splits data into train and test parts
from xgboost import XGBClassifier                # model 2: boosted decision trees

from features import DOMAIN_FEATURE_NAMES        # the exact list of feature columns to use

df = pd.read_csv("data/features.csv")            # table made by prepare.py
X = df[DOMAIN_FEATURE_NAMES]                     # inputs: the feature columns
y = df["is_phishing"]                            # answer to predict: 1 = phishing, 0 = legitimate

# Keep 80% of rows for learning and 20% for testing.
# random_state=42 makes the split repeatable; stratify=y keeps the same phishing share in both parts.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    # 200 trees; min_samples_leaf=3 stops trees memorizing and keeps the file smaller;
    # n_jobs=-1 uses all CPU cores.
    "RandomForest": RandomForestClassifier(
        n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=42
    ),
    # 300 boosting rounds, trees up to depth 6, each round learns at rate 0.1.
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        n_jobs=-1, random_state=42, eval_metric="logloss",
    ),
}

best_name, best_model, best_f1 = None, None, -1   # remember the best model so far

for name, model in models.items():                # try each model in turn
    print(f"\n===== {name} =====")
    model.fit(X_train, y_train)                   # learn from the training data
    preds = model.predict(X_test)                 # predict labels for unseen test data

    print("Accuracy:", round(accuracy_score(y_test, preds), 4))   # share of correct predictions
    # Precision, recall and F1 for each class.
    print(classification_report(y_test, preds, target_names=["legitimate", "phishing"]))
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(y_test, preds))        # counts of right and wrong predictions

    f1 = f1_score(y_test, preds)                  # one score balancing precision and recall
    if f1 > best_f1:                              # keep this model if it beats the best so far
        best_name, best_model, best_f1 = name, model, f1

print(f"\nBest model: {best_name} (phishing F1 = {best_f1:.4f})")

# Which features did the best model rely on most? (watch for one feature dominating)
importances = pd.Series(best_model.feature_importances_, index=DOMAIN_FEATURE_NAMES)
print("\nTop 10 features:\n", importances.sort_values(ascending=False).head(10).round(4))

# Save the model together with its feature list. compress=3 makes the file smaller.
joblib.dump(
    {"model": best_model, "features": DOMAIN_FEATURE_NAMES},
    "models/phishing_model.pkl",
    compress=3,
)
print("\nSaved models/phishing_model.pkl")