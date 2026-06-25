from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
 CORSMiddleware,
 allow_origins=["*"],
 allow_methods=["*"],
 allow_headers=["*"],
)

model = joblib.load('model.pkl')

# Load feature columns if you saved them
try:
 feature_columns = joblib.load('feature_columns.pkl')
except:
 feature_columns = None

class HouseFeatures(BaseModel):
 bedrooms: float
 bathrooms: float
 sqft_living: float
 sqft_lot: float
 floors: float
 waterfront: float
 view: float
 condition: float
 yr_built: float
 yr_renovated: float
 year: float
 month: float
 age: float
 is_renovated: float
 city_mean_price: float

@app.post("/predict")
def predict(data: HouseFeatures):
 # Convert to dict
 try:
    input_dict = data.model_dump()
 except:
    input_dict = data.dict()

 # ✅ FIX 1: Wrap dict in a list to create a single-row DataFrame
 df = pd.DataFrame(input_dict,index=[0])

 # ✅ Reorder columns to match training
 if feature_columns is not None:
    df = df[feature_columns]

 # Predict
 pred = model.predict(df)

 # ✅ FIX 2: pred is an array, use pred[0]
 return {"predicted_price": round(float(pred), 6)*1000000}