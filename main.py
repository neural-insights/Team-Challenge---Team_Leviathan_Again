import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="API de Predicción de Precios de Coches")

# Cargar el modelo
try:
    with open('modelo_coche.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("⚠️ ¡ADVERTENCIA! Modelo no encontrado. La API no funcionará correctamente.")

# Modelo de entrada
class CarFeatures(BaseModel):
    Brand: str
    Model: str
    Year: int
    Engine_Size: float
    Fuel_Type: str
    Transmission: str
    Mileage: int
    Doors: int
    Owner_Count: int

@app.get("/")
async def root():
    return {
        "mensaje": "¡Bienvenido a la API de predicción de precios de coches!",
        "endpoints": {
            "/": "Página de inicio con información general",
            "/predict": "Endpoint POST para predecir el precio de un coche",
            "/docs": "Documentación interactiva automática"
        },
        "ejemplo_predict": {
            "Brand": "Toyota",
            "Model": "Corolla",
            "Year": 2015,
            "Engine_Size": 1.8,
            "Fuel_Type": "Petrol",
            "Transmission": "Manual",
            "Mileage": 50000,
            "Doors": 5,
            "Owner_Count": 2
        }
    }

@app.post("/predict")
async def predict_price(car: CarFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado correctamente")

    car_data = car.model_dump()
    car_df = pd.DataFrame([car_data])
    try:
        prediction = model.predict(car_df)
        price = round(float(prediction[0]), 2)
        return {
            "coche": car_data,
            "precio_predicho": price,
            "mensaje": f"El precio estimado del coche es: {price} euros"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)