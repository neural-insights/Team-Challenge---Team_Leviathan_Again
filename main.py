import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Crear la aplicación FastAPI
app = FastAPI(title="API de Predicción de Precios de Coches")

# Cargar el modelo guardado
try:
    with open('modelo_coche.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    print("¡ADVERTENCIA! Modelo no encontrado. La API no funcionará correctamente.")

# Definir la estructura de datos de entrada


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

# Endpoint raíz - Página de inicio


@app.get("/")
async def root():
    return {
        "mensaje": "¡Bienvenido a la API de predicción de precios de coches!",
        "endpoints": {
            "/": "Esta página de inicio que muestra información de la API",
            "/predict": "Endpoint para predecir el precio de un coche (POST)",
            "/docs": "Documentación automática de la API"
            # El tercer endpoint estará comentado y se habilitará durante la presentación
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

# Endpoint de predicción


@app.post("/predict")
async def predict_price(car: CarFeatures):
    if model is None:
        raise HTTPException(
            status_code=500, detail="Modelo no cargado correctamente")

    # Convertir los datos de entrada en un DataFrame
    car_df = pd.DataFrame([car.dict()])

    # Hacer la predicción
    try:
        prediction = model.predict(car_df)
        # Redondear a 2 decimales y convertir a valor flotante
        price = float(round(prediction[0], 2))

        return {
            "coche": car.dict(),
            "precio_predicho": price,
            "mensaje": f"El precio estimado del coche es: {price} euros"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en la predicción: {str(e)}")

# Tercer endpoint (comentado) - Se habilitará durante la presentación
# @app.get("/health")
# async def health_check():
#     return {
#         "status": "OK",
#         "model_loaded": model is not None,
#         "message": "La API está funcionando correctamente"
#     }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
