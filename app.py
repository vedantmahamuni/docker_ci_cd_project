import joblib
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Get the current project directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

# Load the trained ML model
model = joblib.load(MODEL_PATH)

# Create a FastAPI application
app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict an expected score from study hours.",
    version="1.0.0"
)


# Define input validation
class StudentInput(BaseModel):
    study_hours: float = Field(ge=0, le=24)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Student Performance API is running"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: StudentInput):
    prediction = model.predict([[data.study_hours]])[0]

    # Keep the predicted score between 0 and 100
    prediction = max(0.0, min(100.0, float(prediction)))

    return {
        "study_hours": data.study_hours,
        "predicted_score": round(prediction, 2)
    }