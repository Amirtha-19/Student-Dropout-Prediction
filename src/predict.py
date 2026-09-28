import joblib
import pandas as pd


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

model = joblib.load(
    "models/logistic_regression_model.pkl"
)

print("Model loaded successfully!")


# ============================================================
# 2. GET STUDENT INFORMATION
# ============================================================

print("\n================================")
print("STUDENT DROPOUT RISK PREDICTION")
print("================================\n")


age = int(input("Age: "))

gender = input("Gender (Male/Female): ")

year_of_study = int(
    input("Year of Study (1-4): ")
)

attendance = float(
    input("Attendance Percentage: ")
)

average_grade = float(
    input("Average Grade: ")
)

failed_subjects = int(
    input("Number of Failed Subjects: ")
)

assignments = float(
    input("Assignments Completed Percentage: ")
)

engagement = float(
    input("Engagement Score: ")
)

study_hours = float(
    input("Study Hours Per Day: ")
)

online_learning = float(
    input("Online Learning Hours Per Week: ")
)

family_income = float(
    input("Family Income (INR): ")
)

scholarship = input(
    "Scholarship (Yes/No): "
)

part_time_job = input(
    "Part-Time Job (Yes/No): "
)

financial_stress = float(
    input("Financial Stress Score: ")
)

distance = float(
    input("Distance From College (KM): ")
)

internet_access = input(
    "Internet Access (Yes/No): "
)

parental_support = input(
    "Parental Support (Yes/No): "
)

extracurricular = input(
    "Extracurricular Activities (Yes/No): "
)

disciplinary = input(
    "Disciplinary Issues (Yes/No): "
)


# ============================================================
# 3. CREATE DATAFRAME
# ============================================================

student_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Year_of_Study": [year_of_study],
    "Attendance_Percentage": [attendance],
    "Average_Grade": [average_grade],
    "Failed_Subjects": [failed_subjects],
    "Assignments_Completed_Percentage": [assignments],
    "Engagement_Score": [engagement],
    "Study_Hours_Per_Day": [study_hours],
    "Online_Learning_Hours_Per_Week": [online_learning],
    "Family_Income_INR": [family_income],
    "Scholarship": [scholarship],
    "Part_Time_Job": [part_time_job],
    "Financial_Stress_Score": [financial_stress],
    "Distance_From_College_KM": [distance],
    "Internet_Access": [internet_access],
    "Parental_Support": [parental_support],
    "Extracurricular_Activities": [extracurricular],
    "Disciplinary_Issues": [disciplinary]
})


# ============================================================
# 4. MAKE PREDICTION
# ============================================================

prediction = model.predict(student_data)[0]

probability = model.predict_proba(
    student_data
)[0][1]


# ============================================================
# 5. CONVERT PROBABILITY INTO PERCENTAGE
# ============================================================

dropout_probability = probability * 100


# ============================================================
# 6. DETERMINE RISK LEVEL
# ============================================================

if dropout_probability < 30:

    risk_level = "LOW"

elif dropout_probability < 60:

    risk_level = "MEDIUM"

else:

    risk_level = "HIGH"


# ============================================================
# 7. DISPLAY RESULT
# ============================================================

print("\n================================")
print("PREDICTION RESULT")
print("================================")

print(
    f"Dropout Probability : "
    f"{dropout_probability:.2f}%"
)

print(
    f"Risk Level          : "
    f"{risk_level}"
)

if prediction == 1:

    print(
        "Prediction          : "
        "At Risk of Dropout"
    )

else:

    print(
        "Prediction          : "
        "Not Currently Predicted as Dropout"
    )

print("================================")