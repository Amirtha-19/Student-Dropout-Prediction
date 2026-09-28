import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/student_dataset.csv")

print("Dataset loaded successfully!")
print("Original dataset shape:", df.shape)


# ============================================================
# 2. CHECK DATA TYPES
# ============================================================

print("\n--- Data Types ---")
print(df.dtypes)


# ============================================================
# 3. REMOVE COLUMNS THAT SHOULD NOT BE USED FOR PREDICTION
# ============================================================

# Student_ID       -> identifier, not useful for prediction
# Dropout_Risk_Category -> possible target leakage
# Data_Source      -> only tells us that data is synthetic

columns_to_remove = [
    "Student_ID",
    "Dropout_Risk_Category",
    "Data_Source"
]

df = df.drop(columns=columns_to_remove)

print("\nRemoved columns:")
print(columns_to_remove)

print("\nDataset shape after removing unnecessary columns:")
print(df.shape)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Dropout"])
y = df["Dropout"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 5. AUTOMATICALLY IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


print("\n--- Numerical Features ---")
print(numerical_features)

print("\n--- Categorical Features ---")
print(categorical_features)


# ============================================================
# 6. NUMERICAL PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        # Fill missing numerical values with median
        ("imputer", SimpleImputer(strategy="median")),

        # Standardize numerical values
        ("scaler", StandardScaler())
    ]
)


# ============================================================
# 7. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        # Fill missing categorical values
        ("imputer", SimpleImputer(strategy="most_frequent")),

        # Convert categories into numbers
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ============================================================
# 8. COMBINE BOTH PREPROCESSING PIPELINES
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
# 9. SPLIT DATA INTO TRAINING AND TESTING DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n--- Train/Test Split ---")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 10. FIT PREPROCESSOR ONLY ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

# Apply the already-fitted preprocessor to test data
X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 11. DISPLAY PROCESSED DATA INFORMATION
# ============================================================

print("\n--- Processed Data ---")

print("Processed training data shape:",
      X_train_processed.shape)

print("Processed testing data shape:",
      X_test_processed.shape)


# ============================================================
# 12. CHECK MISSING VALUES AFTER PREPROCESSING
# ============================================================

print("\nPreprocessing completed successfully!")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n==============================")
print("PREPROCESSING SUMMARY")
print("==============================")

print("Original rows:", len(df))

print("Training rows:", len(X_train))

print("Testing rows:", len(X_test))

print("Numerical features:",
      len(numerical_features))

print("Categorical features:",
      len(categorical_features))

print("Final processed training shape:",
      X_train_processed.shape)

print("Final processed testing shape:",
      X_test_processed.shape)

print("==============================")