import pandas as pd
import numpy as np

np.random.seed(42)
n = 600

data = {
    'daily_screen_time_hours': np.random.uniform(1, 16, n).round(1),
    'num_social_media_platforms': np.random.randint(1, 8, n),
    'late_night_usage': np.random.randint(0, 2, n),
    'gpa': np.random.uniform(4.0, 10.0, n).round(2),
    'missed_classes_per_month': np.random.randint(0, 15, n),
    'sleep_hours': np.random.uniform(3, 9, n).round(1),
    'sleep_disturbances': np.random.randint(0, 2, n),
    'physical_activity_hours': np.random.uniform(0, 10, n).round(1),
    'stress_level': np.random.randint(1, 11, n),
    'social_interaction_quality': np.random.randint(1, 6, n),
}

df = pd.DataFrame(data)

def label(row):
    score = (row['daily_screen_time_hours'] * 2
             + row['stress_level']
             + row['missed_classes_per_month'] * 0.5
             + row['late_night_usage'] * 3
             - row['gpa']
             - row['sleep_hours']
             - row['physical_activity_hours'] * 0.5)
    if score < 10:
        return 'Low'
    elif score < 18:
        return 'Moderate'
    elif score < 25:
        return 'High'
    else:
        return 'Severe'

df['addiction_label'] = df.apply(label, axis=1)
df.to_csv('student_data.csv', index=False)
print("Sample data created: student_data.csv")
print(df['addiction_label'].value_counts())
