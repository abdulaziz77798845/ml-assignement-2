"""Train five classifiers and export model pipelines, a labelled test CSV, and metrics."""
import os, json, joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

os.makedirs("model", exist_ok=True)
dataset = load_breast_cancer(as_frame=True)
X, y = dataset.data, dataset.target  # 0=malignant, 1=benign
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
models = {
    "logistic_regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=5000, random_state=42))]),
    "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "knn": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
    "naive_bayes": GaussianNB(),
    "support_vector_machine": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", probability=True, random_state=42))])
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "Accuracy": float(accuracy_score(y_test, pred)),
        "AUC": float(roc_auc_score(y_test, score)),
        "Precision": float(precision_score(y_test, pred, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, zero_division=0)),
        "F1": float(f1_score(y_test, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, pred))
    }
    joblib.dump(model, f"model/{name}.joblib")
pd.concat([X_test, y_test.rename("target")], axis=1).to_csv("test_data.csv", index=False)
with open("metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print(pd.DataFrame(results).T.round(4))
