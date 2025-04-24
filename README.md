# 🚗 API de Predicción de Precios de Coches

API REST construida con **FastAPI** para estimar el precio de un coche según sus características. Usa un modelo de *machine learning* basado en **Random Forest** entrenado con datos reales.

---

## ⚙️ Requisitos

Instala las dependencias:

```bash
pip install -r requirements.txt
```

---

## 🚀 ¿Cómo usar?

1. Entrena el modelo (opcional si ya tienes `modelo_coche.pkl`):

   ```bash
   python entrenamiento_modelo.py
   ```

2. Lanza la API:

   ```bash
   uvicorn main:app --reload
   ```

3. Accede a `http://localhost:8000` o usa `/docs` para probar la API.

---

## 🧪 Ejemplo de entrada

```json
{
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
```

---

## 📦 Endpoints

- `GET /` → Info general y ejemplo
- `POST /predict` → Predicción del precio
- `GET /docs` → Documentación Swagger

---

## 🚹 Autor

**Team Leviathan**