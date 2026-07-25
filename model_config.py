"""Feature schema shared by train.py (offline training) and app.py (serving).

Keeping this in one place means the serving app can never silently drift from
the feature set the exported model artifact was actually trained on.
"""

import numpy as np
import pandas as pd

RANDOM_STATE = 117
DATA_PATH = "sleep_health_dataset.csv"
ARTIFACT_PATH = "models.json"
BUCKETS = ["Low", "Medium", "High"]

PRIMARY_FEATURES = ["age", "stress_score", "steps_that_day", "exercise_day"]

EXTENSION_NUMERIC = [
    "work_hours_that_day", "caffeine_mg_before_bed", "alcohol_units_before_bed",
    "screen_time_before_bed_mins", "bmi", "nap_duration_mins",
]
EXTENSION_BINARY = ["shift_work"]
EXTENSION_CATEGORICAL = [
    "chronotype", "mental_health_condition", "day_type", "season", "gender", "occupation",
]
EXTENSION_RAW_COLUMNS = PRIMARY_FEATURES + EXTENSION_NUMERIC + EXTENSION_BINARY + EXTENSION_CATEGORICAL

CATEGORY_OPTIONS = {
    "chronotype": ["Neutral", "Evening", "Morning"],
    "mental_health_condition": ["Healthy", "Anxiety", "Depression", "Both"],
    "day_type": ["Weekday", "Weekend"],
    "season": ["Spring", "Summer", "Autumn", "Winter"],
    "gender": ["Female", "Male", "Other"],
    "occupation": [
        "Doctor", "Driver", "Freelancer", "Homemaker", "Lawyer", "Manager",
        "Nurse", "Retired", "Sales", "Software Engineer", "Student", "Teacher",
    ],
}


def to_bucket(scores: pd.Series) -> pd.Series:
    rounded = scores.round()
    return pd.cut(rounded, bins=[-np.inf, 4, 6, np.inf], labels=BUCKETS)
