import os
import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, \
    roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tabulate import tabulate
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

load_dotenv()

RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))
IMAGE_DIR = os.getenv("IMAGE_DIR", "images/")
STRATIFIED_K_FOLD = int(os.getenv("STRATIFIED_K_FOLD", 5))
ANALYSIS_DIR = os.getenv("ANALYSIS_DIR", "analysis/")
MODEL_DIR = os.getenv("MODEL_DIR", "models/")
BEST_MODEL_NAME = os.getenv("BEST_MODEL_NAME", "best_model.pkl")
ONE_HOT_ENCODER_NAME = os.getenv("ONE_HOT_ENCODER_NAME", "one_hot_encoder.pkl")
SCALER_NAME = os.getenv("SCALER_NAME", "scaler.pkl")

CATEGORICAL_FEATURES = ["gender", "who_bmi", "depression_severity", "anxiousness", "depressiveness",
                        "sleepiness"]

NUMERIC_FEATURES = ["age", "bmi", "phq_score", "epworth_score"]

DROPPED_FEATURES = ["anxiety_treatment", "anxiety_severity", "depression_treatment", "depression_diagnosis", "suicidal"] # drop the features because it happened after the diagnosis

print("Loading dataset ...")
df = pd.read_csv("data/anxiety_data.csv")

for col in CATEGORICAL_FEATURES:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in NUMERIC_FEATURES:
    df[col] = df[col].fillna(df[col].median())

print(f"Categorical features {', '.join(CATEGORICAL_FEATURES)}")
print(f"Numeric features {', '.join(NUMERIC_FEATURES)}")
print(f"Dropping dropped features {', '.join(DROPPED_FEATURES)}")
df.drop(DROPPED_FEATURES, axis=1, inplace=True)
print("Before dropna:", len(df))
print(df['anxiety_diagnosis'].value_counts())
df.dropna(inplace=True)
print("After dropna:", len(df))
print(df['anxiety_diagnosis'].value_counts())

one_hot_encoder_for_eda = OneHotEncoder(sparse_output=False)
label_encoder = LabelEncoder()

df["anxiety_diagnosis"] = label_encoder.fit_transform(df["anxiety_diagnosis"])

x = df.drop("anxiety_diagnosis", axis=1)
y = df["anxiety_diagnosis"]

print("Number of data :", len(df))
print(f"Target 'anxiety_diagnosis' : {y.unique()}")
# print("Head : ")
# print(df.head())

# print(f"Number of columns is {len(df.columns)} :", df.columns)

print("Encoding categorical features...")

x_categorical = one_hot_encoder_for_eda.fit_transform(df[CATEGORICAL_FEATURES])
x_categorical_features = one_hot_encoder_for_eda.get_feature_names_out(CATEGORICAL_FEATURES)

chi_scores, p_values = chi2(x_categorical, y)

# analyze chi2
chi2_df = pd.DataFrame({
    "feature": x_categorical_features,
    "chi2": chi_scores,
    "p_value": p_values
}).sort_values("chi2", ascending=False)

print("Ensuring interfaces directory exists...")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR, exist_ok=True)

plt.figure(figsize = (12, 8))
plt.barh(chi2_df["feature"], chi2_df["chi2"])
plt.xlabel("Chi-square")
plt.ylabel("Feature")
plt.title("Chi-square test for feature selection")
for index, value in enumerate(chi2_df["chi2"]):
    p_val = chi2_df.iloc[index]["p_value"]
    plt.text(value, index, f" p={p_val:.5f}", va='center')
plt.tight_layout()
plt.savefig(f"{IMAGE_DIR}/feature_importance.png", dpi=300, bbox_inches='tight')
print(f"Plot saved as {IMAGE_DIR}/feature_importance.png")

plt.close()
# plt.show()

# analyze correlation
x_numerical = df[NUMERIC_FEATURES + ["anxiety_diagnosis"]]
correlation = x_numerical.corr()

figure, ax = plt.subplots(figsize = (12, 8))
im = ax.imshow(correlation, vmin=-1, vmax=1)

ax.set_title("Correlation heatmap")
ax.set_xticks(np.arange(len(correlation.columns)), labels=correlation.columns)
ax.set_yticks(np.arange(len(correlation.columns)), labels=correlation.columns)

plt.colorbar(im)
plt.tight_layout()
# plt.show()
plt.savefig(f"{IMAGE_DIR}/correlation.png", dpi=300, bbox_inches='tight')
print(f"Plot saved as {IMAGE_DIR}/correlation.png")

plt.close()

print("Splitting dataset into train and test ...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

def one_hot_encode(x_train, x_test, drop=None, handle_unknown="ignore"):
    one_hot_encoder = OneHotEncoder(drop=drop, sparse_output=False, handle_unknown=handle_unknown)

    return one_hot_encoder.fit_transform(x_train), one_hot_encoder.transform(x_test)

# train all models and analyze the best model
model_lists = [
    (LogisticRegression(max_iter=1000, class_weight='balanced'), 'first'),
    (SVC(kernel='linear', class_weight='balanced', probability=True), 'first'),
    (SGDClassifier(max_iter=1000, class_weight='balanced'), 'first'),

    (CalibratedClassifierCV(RandomForestClassifier(
        max_depth=4,
        min_samples_leaf=10,
        n_estimators=100,
        class_weight='balanced'
    ), method='isotonic'), None),

    (RandomForestClassifier(class_weight='balanced'), None),
    (XGBClassifier(), None)
]

tabulate_body = []
tabulate_headers = ["Model", "Drop", "F1 Score Mean", "F1 Score Std", "F1 Max", "F1 Min"]

def encode_and_merge(x_train, x_test, drop=None, handle_unknown="ignore"):
    one_hot_encoder = OneHotEncoder(drop=drop, sparse_output=False, handle_unknown=handle_unknown)

    x_train_categorical = one_hot_encoder.fit_transform(x_train[CATEGORICAL_FEATURES])
    x_test_categorical = one_hot_encoder.transform(x_test[CATEGORICAL_FEATURES])

    scaler = StandardScaler()
    x_train_numerical = scaler.fit_transform(x_train[NUMERIC_FEATURES])
    x_test_numerical = scaler.transform(x_test[NUMERIC_FEATURES])

    x_train_processed = np.concatenate([x_train_numerical, x_train_categorical], axis=1)
    x_test_processed = np.concatenate([x_test_numerical, x_test_categorical], axis=1)

    return x_train_processed, x_test_processed, one_hot_encoder, scaler


def stratified_k_fold(model, drop, handle_unknown="ignore"):
    skfold = StratifiedKFold(n_splits=STRATIFIED_K_FOLD, random_state=RANDOM_SEED, shuffle=True)

    f1_scores = []

    for train_index, test_index in skfold.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        x_train_processed, x_test_processed, _, _ = encode_and_merge(x_train, x_test, drop, handle_unknown)

        model.fit(x_train_processed, y_train)
        y_pred = model.predict(x_test_processed)

        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)

        print(f"F1 score for k fold {len(f1_scores)} : {f1}")

    mean = float(np.mean(f1_scores))
    std  = float(np.std(f1_scores))

    print(f"F1 score mean for {model.__class__.__name__}: {mean}")
    print(f"F1 score std for {model.__class__.__name__}: {std}")

    tabulate_body.append([model.__class__.__name__, drop if drop is not None else 'None', mean, std, mean + std, mean - std])

def train_model():
    for model, drop in model_lists:
        print(f"Run stratified k fold for [{model.__class__.__name__}] ...")

        stratified_k_fold(model, drop)

    print(f"Ensure analysis directory exists...")
    if not os.path.exists(ANALYSIS_DIR):
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
    print(f"Save analysis to {ANALYSIS_DIR}/analysis.csv")
    df_analysis = pd.DataFrame(tabulate_body, columns=tabulate_headers)
    df_analysis.to_csv(f"{ANALYSIS_DIR}/analysis.csv")
    print(f"Analysis saved to {ANALYSIS_DIR}/analysis.csv")

train_model()

print("Tabulate : ")
print(tabulate(tabulate_body, headers=tabulate_headers, tablefmt='fancy_grid', showindex=False, numalign="center"))

chosen_models = [
    # (RandomForestClassifier(class_weight='balanced'), None),
    (CalibratedClassifierCV(RandomForestClassifier(
        max_depth=4,
        min_samples_leaf=10,
        n_estimators=100,
        class_weight='balanced'
    ), method='isotonic'), None),
    # (XGBClassifier(), None)
]

def run_chosen_model():
    for model, drop in chosen_models:
        x_train_processed, x_test_processed, one_hot_encoder, scaler = encode_and_merge(x_train, x_test, drop, 'ignore')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_train_smote, y_train_smote = SMOTE().fit_resample(x_train_processed, y_train)

        model.fit(x_train_smote, y_train_smote)
        y_pred = model.predict(x_test_processed)

        print(f"Running metrics for [{model.__class__.__name__}] ...")
        print(f"Classification report for [{model.__class__.__name__}] :")
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        print(cr)

        y_proba = model.predict_proba(x_test_processed)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

        f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

        best_threshold = thresholds[np.argmax(f1)]

        print("Threshold :", best_threshold)

        y_pred = (y_proba >= best_threshold).astype(int)

        pos_probs = y_proba[y_test == 1]

        low = np.percentile(pos_probs, 33)
        high = np.percentile(pos_probs, 66)

        y_proba = model.predict_proba(x_test_processed)[:, 1]

        print("Low boundary:", low)
        print("High boundary:", high)

        sample = x_train[y_train == 1].iloc[0]
        sample_cat = one_hot_encoder.transform(pd.DataFrame([sample[CATEGORICAL_FEATURES]]))
        sample_num = scaler.transform(pd.DataFrame([sample[NUMERIC_FEATURES]]))
        sample_features = np.concatenate([sample_num, sample_cat], axis=1)
        prob = model.predict_proba(sample_features)[0][1]
        print(f"Prob training anxiety: {prob:.3f}")

        config = {
            "threshold": float(best_threshold),
            "low_boundary": float(np.percentile(y_proba, 33)),
            "high_boundary": float(np.percentile(y_proba, 66))
        }

        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(f"{MODEL_DIR}/config.pkl", "wb") as f:
            pickle.dump(config, f)
        print(f"Config saved to {MODEL_DIR}/config.pkl")

        print(f"Ensuring model {MODEL_DIR}/ exists.")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR, exist_ok=True)
        print(f"Saving best model to {MODEL_DIR}/{BEST_MODEL_NAME}...")
        with open(f"{MODEL_DIR}/{BEST_MODEL_NAME}", "wb") as f:
            pickle.dump(model, f)
        print(f"Best model saved to {MODEL_DIR}/{BEST_MODEL_NAME}")

        print(f"Saving one hot encoder to {MODEL_DIR}/{ONE_HOT_ENCODER_NAME}...")
        with open(f"{MODEL_DIR}/{ONE_HOT_ENCODER_NAME}", "wb") as f:
            pickle.dump(one_hot_encoder, f)
        print(f"One hot encoder saved to {MODEL_DIR}/{ONE_HOT_ENCODER_NAME}")

        print(f"Saving scaler to {MODEL_DIR}/{SCALER_NAME}...")
        with open(f"{MODEL_DIR}/{SCALER_NAME}", "wb") as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to {MODEL_DIR}/{SCALER_NAME}")

        if hasattr(model, 'feature_importances_'):
            importance_features = pd.DataFrame({
                'feature': list(NUMERIC_FEATURES) + list(one_hot_encoder.get_feature_names_out(CATEGORICAL_FEATURES)),
                'importance': model.feature_importances_
            }).sort_values(by='importance', ascending=False)

            print(f"Ensure analysis directory exists...")
            if not os.path.exists(ANALYSIS_DIR):
                os.makedirs(ANALYSIS_DIR, exist_ok=True)

            print(f"Saving feature importance to {ANALYSIS_DIR}/{model.__class__.__name__}_feature_importance.csv")
            importance_features.to_csv(f"{ANALYSIS_DIR}/{model.__class__.__name__}_feature_importance.csv", index=False)
            print(f"Feature importance saved to {ANALYSIS_DIR}/{model.__class__.__name__}_feature_importance.csv")
        else:
            print(f"Model {model.__class__.__name__} does not have feature importance. Skipped.")

        # print(f"Feature importance for {model.__class__.__name__} is :")
        # print(importance_features.head(10))

        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_test.unique())
        display.plot()

        plt.savefig(f"{IMAGE_DIR}/{model.__class__.__name__}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved to {IMAGE_DIR}/{model.__class__.__name__}_confusion_matrix.png")

        y_proba = model.predict_proba(x_test_processed)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC for [{model.__class__.__name__}] : {roc_auc}")

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        print(f"Saving ROC curve for [{model.__class__.__name__}] ...")
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle='--')  # garis random
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig(f"{IMAGE_DIR}/{model.__class__.__name__}_roc_curve.png", dpi=300)
        plt.close()
        print(f"ROC curve saved to {IMAGE_DIR}/{model.__class__.__name__}_roc_curve.png")

print("Running chosen model...")
run_chosen_model()