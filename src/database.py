import sqlite3

DATABASE_NAME = "student_predictions.db"


def create_database():

    connection = sqlite3.connect(DATABASE_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            age INTEGER,
            gender TEXT,
            year_of_study INTEGER,

            attendance_percentage REAL,
            average_grade REAL,
            failed_subjects INTEGER,
            assignments_completed_percentage REAL,

            engagement_score REAL,
            study_hours_per_day REAL,
            online_learning_hours_per_week REAL,

            family_income_inr REAL,
            scholarship TEXT,
            part_time_job TEXT,

            financial_stress_score REAL,
            distance_from_college_km REAL,

            internet_access TEXT,
            parental_support TEXT,
            extracurricular_activities TEXT,
            disciplinary_issues TEXT,

            dropout_probability REAL,
            risk_level TEXT,
            prediction TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()

    print("Database created successfully!")


if __name__ == "__main__":
    create_database()