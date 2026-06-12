"""
generate_google_form_sample.py
───────────────────────────────
Generates a realistic mock Google Form CSV export
(with full question-text column headers) for testing the teacher dashboard.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(99)
n = 40  # 40 students — realistic class size

names = [
    "Arpita Pradhane","Mouneshwar Sutar","Parasuram Sanadi","Pratiksha Pawar",
    "Rohit Kumbhar","Sneha Patil","Akash Desai","Priya Naik","Vijay More",
    "Deepa Kulkarni","Suresh Yadav","Anita Sharma","Ravi Kumar","Meena Joshi",
    "Arun Patil","Kavita Singh","Nikhil Gupta","Pooja Reddy","Sanjay Nair",
    "Lata Verma","Rajesh Bhat","Sunita Rao","Mahesh Iyer","Usha Pillai",
    "Ganesh Hedge","Rekha Kadam","Arjun Pawar","Neha Gaikwad","Kiran Salvi",
    "Mamta Wagh","Dinesh Lokhande","Alka Thorat","Vishal Mane","Shital Bhosale",
    "Prasad Jadhav","Vrushali Patil","Omkar Chavan","Madhuri Shirke",
    "Tejas Kulkarni","Swati Deshmukh"
]

depts    = ['CSE']*20 + ['ECE']*10 + ['MECH']*5 + ['CIVIL']*5
semesters= np.random.choice([3,4,5,6], n)
usns     = [f"2KD23CS{str(i+1).zfill(3)}" for i in range(n)]

screen_times = np.clip(np.random.normal(7, 3, n), 1, 16).round(1)
social_media = np.random.randint(1, 7, n)
late_night   = np.random.choice(['Yes','No'], n, p=[0.55, 0.45])
gpa          = np.clip(np.random.normal(7.0, 1.5, n), 4.0, 10.0).round(2)
missed       = np.clip(np.random.poisson(3, n), 0, 14).astype(int)
sleep_hrs    = np.clip(np.random.normal(6.0, 1.2, n), 3.0, 9.0).round(1)
sleep_dist   = np.random.choice(['Yes','No'], n, p=[0.45, 0.55])
phys_act     = np.clip(np.random.exponential(2.5, n), 0, 10).round(1)
stress       = np.random.randint(1, 11, n)
social_qual  = np.random.randint(1, 6, n)

import datetime
base_time = datetime.datetime(2025, 3, 1, 9, 0, 0)
timestamps = [base_time + datetime.timedelta(minutes=int(i*7 + np.random.randint(0,5)))
              for i in range(n)]

df = pd.DataFrame({
    'Timestamp': [t.strftime('%Y/%m/%d %H:%M:%S') for t in timestamps],
    'What is your full name?'        : names[:n],
    'What is your student ID / USN?' : usns,
    'What is your department / branch?': depts,
    'What is your semester?'         : semesters,

    'How many hours per day do you use electronic gadgets (phone/tablet/laptop/gaming)?':
        screen_times,
    'How many social media platforms do you actively use?':
        social_media,
    'Do you use gadgets late at night (after 10 PM)?':
        late_night,

    'What is your current GPA / CGPA?': gpa,
    'How many classes have you missed in the past month?': missed,

    'How many hours do you sleep per night on average?': sleep_hrs,
    'Do you experience sleep disturbances (difficulty sleeping, waking at night)?':
        sleep_dist,
    'How many hours per week do you spend on physical activity / exercise?':
        phys_act,

    'On a scale of 1–10, how would you rate your current stress level?': stress,
    'On a scale of 1–5, how would you rate the quality of your social interactions?':
        social_qual,
})

out_dir  = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, 'sample_google_form_responses.csv')
df.to_csv(out_path, index=False)

print(f"Sample Google Form CSV created: {out_path}")
print(f"Shape: {df.shape}")
print("\nColumn headers (as Google Forms exports them):")
for col in df.columns:
    print(f"  {col}")
