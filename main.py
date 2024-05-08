import uvicorn
import pickle
import json

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()
OPENAI_API_KEY = "sk-proj-0msCPUB1tRGcLbQUbQoxT3BlbkFJvqYddOIoJF5OMIrdTHgN"

client = OpenAI(api_key=OPENAI_API_KEY)
# initialize open ai api client

origins = ["http://localhost"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = pickle.load(open('cc.pkl', 'rb'));


# define model input types
class Patient(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# setting up home route
@app.get("/")
def read_root():
    return {"data": "Welcome to the Integrated Framework for Prediction"}


# setup prediction route
@app.post("/prediction/")
async def get_prediction(data: Patient):
    sample = [[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]]
    result = model.predict(sample).tolist()[0]
    # Additional information
    probability_estimates = None
    confidence_scores = None
    feature_importance = None

    # If the model supports probability estimates
    if hasattr(model, 'predict_proba'):
        probability_estimates = model.predict_proba(sample).tolist()[0]

    # If the model supports decision function or predict_proba for confidence scores
    if hasattr(model, 'decision_function') or hasattr(model, 'predict_proba'):
        confidence_scores = model.decision_function(sample).tolist()[0] if hasattr(model,
                                                                                   'decision_function') else probability_estimates

    # If the model supports feature importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_.tolist()
    prompt = ("Based on your health data and diabetes prediction, " +
              "here are some personalized recommendations: " +
              f"Age: {data.Age}, "
              f"BMI: {data.BMI}, Blood Pressure: {data.BloodPressure}, "
              f"Insulin: {data.Insulin}, Skin Thickness: {data.SkinThickness}, "
              f"Glucose: {data.Glucose}. "
              "You should...")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    recommendations = json.loads(response.choices[0].message.content)["recommendations"]
    # Prepare response
    response_data = {
        "prediction": result,
        "recommendations": recommendations,
        "probability_estimates": probability_estimates,
        "confidence_scores": confidence_scores,
        "feature_importance": feature_importance
    }
    return {"data": {"prediction": result, "meta": response_data}, "recommendations": response.choices[0].message.content}



# configure server and host
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')
