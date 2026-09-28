import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/student_dataset.csv")

# 1. Dataset size
print("Dataset shape:")
print(df.shape)

# 2. Column names
print("\nColumns:")
print(df.columns.tolist())

# 3. First 5 rows
print("\nFirst 5 rows:")
print(df.head())

# 4. Missing values
print("\nMissing values:")
print(df.isnull().sum())

# 5. Target distribution
print("\nDropout distribution:")
print(df["Dropout"].value_counts())

# 6. Dropout percentage
print("\nDropout percentage:")
print(df["Dropout"].value_counts(normalize=True) * 100)

# Dropout distribution visualization

dropout_counts = df["Dropout"].value_counts()

plt.bar(["No Dropout", "Dropout"], dropout_counts)

plt.title("Student Dropout Distribution")
plt.xlabel("Dropout Status")
plt.ylabel("Number of Students")


# Attendance vs Dropout

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df[df["Dropout"] == 0]["Attendance_Percentage"].dropna(),
        df[df["Dropout"] == 1]["Attendance_Percentage"].dropna()
    ],
    labels=["No Dropout", "Dropout"]
)

plt.title("Attendance Percentage vs Dropout")
plt.xlabel("Dropout Status")
plt.ylabel("Attendance Percentage")

# Average Grade vs Dropout

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df[df["Dropout"] == 0]["Average_Grade"].dropna(),
        df[df["Dropout"] == 1]["Average_Grade"].dropna()
    ],
    labels=["No Dropout", "Dropout"]
)

plt.title("Average Grade vs Dropout")
plt.xlabel("Dropout Status")
plt.ylabel("Average Grade")

# Engagement Score vs Dropout

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df[df["Dropout"] == 0]["Engagement_Score"].dropna(),
        df[df["Dropout"] == 1]["Engagement_Score"].dropna()
    ],
    labels=["No Dropout", "Dropout"]
)

plt.title("Engagement Score vs Dropout")
plt.xlabel("Dropout Status")
plt.ylabel("Engagement Score")

# Financial Stress vs Dropout

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df[df["Dropout"] == 0]["Financial_Stress_Score"].dropna(),
        df[df["Dropout"] == 1]["Financial_Stress_Score"].dropna()
    ],
    labels=["No Dropout", "Dropout"]
)

plt.title("Financial Stress Score vs Dropout")
plt.xlabel("Dropout Status")
plt.ylabel("Financial Stress Score")

plt.show()

# ==========================================
# STEP 4: DATA QUALITY & LEAKAGE CHECK
# ==========================================

print("\n--- Dropout by Risk Category ---")

print(
    pd.crosstab(
        df["Dropout_Risk_Category"],
        df["Dropout"],
        normalize="index"
    ) * 100
)

print("\n--- Average values by Dropout ---")

print(
    df.groupby("Dropout")[
        [
            "Attendance_Percentage",
            "Average_Grade",
            "Failed_Subjects",
            "Assignments_Completed_Percentage",
            "Engagement_Score",
            "Study_Hours_Per_Day",
            "Financial_Stress_Score"
        ]
    ].mean()
)