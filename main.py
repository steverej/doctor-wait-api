import pandas as pd
import datetime
import re
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Initialize FastAPI
app = FastAPI(title="Doctor Wait Time Predictor API")

# ----------------- Load and Train Model Once -----------------
try:
    df = pd.read_csv('doctor_behaviour_updated.csv')

    # Feature Engineering
    df['TimeTaken_seconds'] = pd.to_timedelta(df['TimeTaken']).dt.total_seconds()
    df['AvgWaitTime_seconds'] = pd.to_timedelta(df['AvgWaitTime']).dt.total_seconds()

    # Filter > 30 mins (adjust threshold if needed)
    df = df[df['AvgWaitTime_seconds'] <= (30 * 60)]

    # Extract hour and minute
    time_called_dt = pd.to_datetime(df['TimeCalled'].astype(str), format='%H:%M:%S', errors='coerce')
    df['hour'] = time_called_dt.dt.hour
    df['minute'] = time_called_dt.dt.minute

    # Token number
    df['token_num'] = df['tokenNumber'].astype(str).str.extract(r'(\d+)').astype(float)

    # Boolean flag
    df['served_after_break'] = df['servedAfterBreak'].astype(int)

    # Drop NaNs
    df.dropna(subset=['hour', 'minute', 'token_num', 'AvgWaitTime_seconds'], inplace=True)

    # Features and target
    features = ['hour', 'minute', 'token_num', 'served_after_break', 'type', 'day']
    target = 'AvgWaitTime_seconds'

    X = df[features]
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = ['type', 'day']
    numeric_features = ['hour', 'minute', 'token_num', 'served_after_break']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

    pipeline.fit(X_train, y_train)

    # Global average wait time
    overall_avg_wait_seconds = df['AvgWaitTime_seconds'].mean()

except Exception as e:
    print(f"Error during setup: {e}")
    overall_avg_wait_seconds = 600  # default 10 min


# ----------------- API Models -----------------
class DoctorInfo(BaseModel):
    doctor_id: str
    specialization: str
    day: str
    start_hour: int
    start_minute: int
    token_number: str


# ----------------- API Endpoints -----------------

@app.get("/")
def root():
    return {"message": "Doctor Wait Time Predictor API is running 🚀"}


@app.get("/average_wait_time")
def get_average_wait_time():
    """Returns overall average wait time from dataset"""
    minutes = int(overall_avg_wait_seconds // 60)
    seconds = int(overall_avg_wait_seconds % 60)
    return {
        "average_wait_time_seconds": overall_avg_wait_seconds,
        "formatted": f"{minutes} minutes {seconds} seconds"
    }


@app.post("/predict_token_time")
def predict_token(info: DoctorInfo):
    """Predicts cumulative waiting time & estimated appointment for a token"""

    # Extract token number
    match = re.search(r'(\d+)', info.token_number)
    if not match:
        return {"error": "Invalid token format. Use like A1, W7, or 12."}

    token_number = int(match.group(1))
    avg_sec = overall_avg_wait_seconds
    total_elapsed_seconds = token_number * avg_sec

    estimated_wait_timedelta = datetime.timedelta(seconds=total_elapsed_seconds)
    today = datetime.date.today()
    doctor_start_datetime = datetime.datetime(
        today.year, today.month, today.day, info.start_hour, info.start_minute, 0
    )
    estimated_appointment_time = doctor_start_datetime + estimated_wait_timedelta

    wait_minutes = int(total_elapsed_seconds // 60)
    wait_seconds = int(total_elapsed_seconds % 60)

    return {
        "doctor_id": info.doctor_id,
        "specialization": info.specialization,
        "day": info.day,
        "token_number": info.token_number,
        "estimated_cumulative_wait": f"{wait_minutes} minutes {wait_seconds} seconds",
        "estimated_appointment_time": estimated_appointment_time.strftime("%I:%M %p")
    }
