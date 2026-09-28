import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/student_dataset.csv")

print("Dataset loaded successfully!")


# ============================================================
# 2. REMOVE UNNECESSARY / LEAKAGE COLUMNS
# ============================================================

df = df.drop(
    columns=[
        "Student_ID",
        "Dropout_Risk_Category",
        "Data_Source"
    ]
)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Dropout"])
y = df["Dropout"]


print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True) * 100)


# ============================================================
# 4. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 5. NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ============================================================
# 6. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ============================================================
# 7. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n================================")
print("TRAIN / TEST SPLIT")
print("================================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# ============================================================
# 10. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

logistic_model.fit(X_train, y_train)

print("Logistic Regression training completed!")

joblib.dump(
    logistic_model,
    "models/logistic_regression_model.pkl"
)

print("\nLogistic Regression model saved successfully!")


# ============================================================
# 11. LOGISTIC REGRESSION PREDICTIONS
# ============================================================

logistic_predictions = logistic_model.predict(X_test)


# ============================================================
# 12. LOGISTIC REGRESSION PROBABILITIES
# ============================================================

logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]


# ============================================================
# 13. LOGISTIC REGRESSION METRICS
# ============================================================

logistic_accuracy = accuracy_score(
    y_test,
    logistic_predictions
)

logistic_precision = precision_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_recall = recall_score(
    y_test,
    logistic_predictions,
    zero_division=0
)

logistic_f1 = f1_score(
    y_test,
    logistic_predictions,
    zero_division=0
)


print("\n================================")
print("LOGISTIC REGRESSION RESULTS")
print("================================")

print("Accuracy :", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("F1 Score :", round(logistic_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logistic_predictions,
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        logistic_predictions
    )
)


# ============================================================
# 14. RANDOM FOREST
# ============================================================

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# ============================================================
# 15. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

random_forest_model.fit(X_train, y_train)

print("Random Forest training completed!")


# ============================================================
# 16. RANDOM FOREST PREDICTIONS
# ============================================================

random_forest_predictions = random_forest_model.predict(X_test)


# ============================================================
# 17. RANDOM FOREST PROBABILITIES
# ============================================================

random_forest_probabilities = (
    random_forest_model.predict_proba(X_test)[:, 1]
)


# ============================================================
# 18. RANDOM FOREST METRICS
# ============================================================

rf_accuracy = accuracy_score(
    y_test,
    random_forest_predictions
)

rf_precision = precision_score(
    y_test,
    random_forest_predictions,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    random_forest_predictions,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    random_forest_predictions,
    zero_division=0
)


print("\n================================")
print("RANDOM FOREST RESULTS")
print("================================")

print("Accuracy :", round(rf_accuracy, 4))
print("Precision:", round(rf_precision, 4))
print("Recall   :", round(rf_recall, 4))
print("F1 Score :", round(rf_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        random_forest_predictions,
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        random_forest_predictions
    )
)


# ============================================================
# 19. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest"
        ],

        "Accuracy": [
            logistic_accuracy,
            rf_accuracy
        ],

        "Precision": [
            logistic_precision,
            rf_precision
        ],

        "Recall": [
            logistic_recall,
            rf_recall
        ],

        "F1 Score": [
            logistic_f1,
            rf_f1
        ]
    }
)


# ============================================================
# 20. DISPLAY MODEL COMPARISON
# ============================================================

print("\n================================")
print("MODEL COMPARISON")
print("================================")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 21. SAVE MODEL COMPARISON
# ============================================================

comparison.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nModel comparison saved as:")
print("model_comparison.csv")


# ============================================================
# 22. PROBABILITY SAMPLE
# ============================================================

probability_results = pd.DataFrame(
    {
        "Actual_Dropout": y_test.values,
        "Logistic_Probability": logistic_probabilities,
        "Random_Forest_Probability": random_forest_probabilities
    }
)


print("\n================================")
print("SAMPLE DROPOUT PROBABILITIES")
print("================================")

print(
    probability_results.head(10).to_string(
        index=False
    )
)


print("\n================================")
print("TRAINING COMPLETED")
print("================================")