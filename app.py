from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn


app = FastAPI()

# Load your trained model, label encoder, and vectorizer
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class Report(BaseModel):
    text: str

# CORS middleware to allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(report: Report):
    vectorized_text = vectorizer.transform([report.text])
    prediction = model.predict(vectorized_text)
    label = label_encoder.inverse_transform(prediction)
    return {"prediction": label[0]}

def main():
    uvicorn.run(app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()
