"""
Smart Robbery and Theft from the Person Risk Analysis Using Machine Learning
Full training pipeline - Chicago Police Department data (2018)

"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay, ConfusionMatrixDisplay
)
import joblib

sns.set_theme(style="whitegrid", font_scale=1.0)
FIG = "figures"
MODEL_DIR = "model"
DATA_DIR = "data"
RESULTS = {}

# ---------------------------------------------------------------------------
# 1. LOAD REAL DATA 
# ---------------------------------------------------------------------------
df_raw = pd.read_csv(f"{DATA_DIR}/crimes_2018.csv")
print("Raw shape:", df_raw.shape)
RESULTS["raw_records"] = int(df_raw.shape[0])

# ---------------------------------------------------------------------------
# 2. FILTER to Robbery + Theft-from-the-person (Pocket-picking, Purse-snatching)
# ---------------------------------------------------------------------------
mask = (df_raw["Primary Type"] == "ROBBERY") | (
    (df_raw["Primary Type"] == "THEFT") & (df_raw["Description"].isin(["POCKET-PICKING", "PURSE-SNATCHING"]))
)
df = df_raw[mask].copy()
df = df.dropna(subset=["Latitude", "Longitude", "Location Description"])
print("Filtered shape:", df.shape)
print(df["Primary Type"].value_counts())
RESULTS["filtered_records"] = int(df.shape[0])
RESULTS["class_counts"] = df["Primary Type"].value_counts().to_dict()

# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["Date_dt"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p")
df["Hour"] = df["Date_dt"].dt.hour
df["Month"] = df["Date_dt"].dt.month
df["DayOfWeek"] = df["Date_dt"].dt.dayofweek  

le_loc = LabelEncoder()
df["LocDesc_enc"] = le_loc.fit_transform(df["Location Description"].astype(str))
joblib.dump(le_loc, f"{MODEL_DIR}/location_encoder.joblib")
RESULTS["n_location_types"] = int(len(le_loc.classes_))

le_target = LabelEncoder()
df["target"] = le_target.fit_transform(df["Primary Type"])
joblib.dump(le_target, f"{MODEL_DIR}/label_encoder.joblib")
print("Label encoding:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))
RESULTS["label_encoding"] = dict(zip(le_target.classes_, [int(x) for x in le_target.transform(le_target.classes_)]))


FEATURES = ["Latitude", "Longitude", "Hour", "Month", "DayOfWeek", "LocDesc_enc", "District", "Ward", "Community Area"]
df = df.dropna(subset=FEATURES)
print("Final modelling shape:", df.shape)
RESULTS["final_records"] = int(df.shape[0])

df.to_csv(f"{DATA_DIR}/feature_engineered_dataset.csv", index=False)

# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
sns.countplot(x="Primary Type", data=df, ax=axes[0], palette=["#c0392b", "#2c7a4b"])
axes[0].set_title("Crime Type Distribution")
for p in axes[0].patches:
    axes[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width()/2, p.get_height()),
                      ha="center", va="bottom", fontsize=9)

sns.histplot(data=df, x="Hour", hue="Primary Type", bins=24, multiple="dodge",
             palette={"ROBBERY": "#c0392b", "THEFT": "#2c7a4b"}, ax=axes[1])
axes[1].set_title("Hour-of-Day Distribution by Crime Type")

top_loc = df["Location Description"].value_counts().head(10)[::-1]
axes[2].barh(top_loc.index, top_loc.values, color="#8e44ad")
axes[2].set_title("Top 10 Locations by Incident Count")
plt.tight_layout()
plt.savefig(f"{FIG}/01_eda_overview.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
X = df[FEATURES]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
RESULTS["split"] = {"n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0])}

# ---------------------------------------------------------------------------
# 5. LOGISTIC REGRESSION BASELINE
# ---------------------------------------------------------------------------
lr_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
lr_pipe.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr_pipe.predict(X_test))
print(f"Logistic Regression baseline test accuracy: {lr_acc:.4f}")
RESULTS["baseline_lr_accuracy"] = float(lr_acc)

fig, ax = plt.subplots(figsize=(5, 4.5))
cm_lr = confusion_matrix(y_test, lr_pipe.predict(X_test))
ConfusionMatrixDisplay(cm_lr, display_labels=le_target.classes_).plot(ax=ax, cmap="Purples", colorbar=False)
ax.set_title("Confusion Matrix - Logistic Regression Baseline")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{FIG}/02_baseline_confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. FAIR, UNIFORM MODEL COMPARISON - 
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_configs = {
    "LogisticRegression": (
        Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        {"clf__C": [0.1, 1, 10]},
    ),
    "DecisionTree": (
        Pipeline([("scaler", StandardScaler()), ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42))]),
        {"clf__max_depth": [6, 10, 15, None], "clf__min_samples_leaf": [1, 5]},
    ),
    "RandomForest": (
        Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))]),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [10, 15, None]},
    ),
    "GradientBoosting": (
        Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(random_state=42))]),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 4, 5]},
    ),
    "KNN": (
        Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
        {"clf__n_neighbors": [5, 15, 25], "clf__weights": ["uniform", "distance"]},
    ),
}

tuned_results = {}
best_estimators = {}
for name, (pipe, grid) in model_configs.items():
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_weighted", n_jobs=1, error_score="raise")
    gs.fit(X_train, y_train)
    tuned_results[name] = {"best_params": gs.best_params_, "best_cv_f1_weighted": gs.best_score_}
    best_estimators[name] = gs.best_estimator_
    print(f"{name:20s} -> {gs.best_params_}  CV F1(weighted) = {gs.best_score_:.4f}")

RESULTS["tuning"] = tuned_results
with open(f"{FIG}/results_partial.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

tuned_df = pd.DataFrame([{"model": k, "cv_f1_weighted": v["best_cv_f1_weighted"]} for k, v in tuned_results.items()])
tuned_df = tuned_df.sort_values("cv_f1_weighted", ascending=False).reset_index(drop=True)
tuned_df.to_csv(f"{FIG}/tuning_results.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=tuned_df, x="model", y="cv_f1_weighted", ax=ax, palette="viridis")
ax.set_title("Tuned Cross-Validated F1 (weighted) by Model\n(uniform search protocol, full training set)")
ax.set_ylim(0.5, min(1.0, tuned_df["cv_f1_weighted"].max() + 0.08))
for i, v in enumerate(tuned_df["cv_f1_weighted"]):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{FIG}/03_model_comparison_tuned.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. FINAL MODEL: refit winner, evaluate once on held-out test set
# ---------------------------------------------------------------------------
final_name = tuned_df.iloc[0]["model"]
final_params = tuned_results[final_name]["best_params"]
print(f"\nSELECTED FINAL MODEL: {final_name}  params={final_params}")
RESULTS["final_model_selected"] = final_name
RESULTS["final_model_params"] = final_params

base_pipe, _ = model_configs[final_name]
final_pipe = base_pipe.set_params(**final_params)
final_pipe.fit(X_train, y_train)

y_pred = final_pipe.predict(X_test)
y_score = final_pipe.predict_proba(X_test)[:, 1] if hasattr(final_pipe.named_steps["clf"], "predict_proba") else None

test_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
    "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
    "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
}
if y_score is not None:
    test_metrics["roc_auc"] = roc_auc_score(y_test, y_score)
print("HELD-OUT TEST METRICS:", test_metrics)
RESULTS["test_metrics"] = test_metrics
RESULTS["classification_report"] = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)

fig, ax = plt.subplots(figsize=(5, 4.5))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=le_target.classes_).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix - {final_name} (Held-out Test Set)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{FIG}/04_final_confusion_matrix.png", dpi=150)
plt.close()

if y_score is not None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y_test, y_score, ax=ax, name=final_name, color="#2c3e50")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(f"ROC Curve - {final_name} (Test Set)")
    plt.tight_layout()
    plt.savefig(f"{FIG}/05_roc_curve.png", dpi=150)
    plt.close()

clf_final = final_pipe.named_steps["clf"]
if hasattr(clf_final, "feature_importances_"):
    imp = pd.Series(clf_final.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    imp.plot.barh(ax=ax, color="#16a085")
    ax.set_title(f"Feature Importance - {final_name}")
    plt.tight_layout()
    plt.savefig(f"{FIG}/06_feature_importance.png", dpi=150)
    plt.close()
    RESULTS["feature_importance"] = imp.to_dict()

joblib.dump(final_pipe, f"{MODEL_DIR}/crime_pipeline.joblib")
with open(f"{FIG}/results.json", "w") as f:
    json.dump(RESULTS, f, indent=2, default=str)

print("\nDONE. Final model:", final_name, "saved to", f"{MODEL_DIR}/crime_pipeline.joblib")
print("Test metrics:", test_metrics)
