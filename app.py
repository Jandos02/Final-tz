from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Optional
import json

app = FastAPI(
    title="Bank Churn Prediction API",
    description="API для прогнозирования оттока клиентов банка",
    version="1.0.0",
)


# Определяем pydantic модель данных для входящих запросов
class ClientData(BaseModel):
    кредитный_рейтинг: float
    город: str
    пол: str
    возраст: float
    стаж_в_банке: float
    баланс_депозита: Optional[float] = None
    число_продуктов: float
    есть_кредитка: float
    активный_клиент: float
    оценочная_зарплата: float


# Pydantic Модель для ответа
class PredictionResponse(BaseModel):
    score: float
    prediction: int


# Глобальные переменные для моделей
model = None
label_encoder_gender = None
columns_info = None


@app.on_event("startup")
async def load_models():
    """Загрузка моделей при запуске сервиса"""
    global model, label_encoder_gender, columns_info

    try:
        model = joblib.load("model/final_model.joblib")
        print("Модель загружена")

        label_encoder_gender = joblib.load("model/label_encoder_gender.joblib")
        print("Энкодер для пола загружен")

        with open("model/columns_info.json", "r", encoding="utf-8") as f:
            columns_info = json.load(f)
        print("Информация о столбцах загружена")

    except Exception as e:
        print(f"Ошибка при загрузке моделей: {e}")
        raise e


def preprocess_data(client_data: ClientData) -> pd.DataFrame:
    """Предобработка входных данных"""

    data = pd.DataFrame([client_data.dict()])

    # Обрабатываем пропущенные значения (заполняем медианой)
    if data["баланс_депозита"].isna().any():
        data["баланс_депозита"].fillna(
            122570.69, inplace=True
        )  # медиана из обучающих данных (медиана была взята из обучающей выборки)

    # Кодируем пол
    data["пол"] = label_encoder_gender.transform(data["пол"])

    # One-hot encoding для города
    city_dummies = pd.get_dummies(data["город"], prefix="город")
    data = data.drop("город", axis=1)
    data = pd.concat([data, city_dummies], axis=1)

    model_params = columns_info["feature_columns"]
    for col in model_params:
        if col not in data.columns:
            data[col] = 0  # для недостающих городов добавляем столбец с нулями

    # Переупорядочиваем столбцы в том же порядке, что и при обучении
    data = data[model_params]

    return data


@app.get("/")
async def root():
    """Главная страница API"""
    return {
        "message": "Bank Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": columns_info["model_type"] if columns_info else "unknown",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(client_data: ClientData):
    """
    Прогнозирование оттока клиента

    Принимает данные клиента и возвращает:
    - score: вероятность оттока (0-1)
    - prediction: бинарный прогноз (0 - остается, 1 - уйдет)
    """

    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        # Предобрабатываем данные
        processed_data = preprocess_data(client_data)

        # Получаем предсказания
        probability = model.predict_proba(processed_data)[0, 1]
        prediction = int(probability > 0.5)

        return PredictionResponse(
            score=float(probability),
            prediction=prediction,
        )

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Ошибка при обработке данных: {str(e)}"
        )


@app.get("/model_info")
async def get_model_info():
    """Информация о модели"""
    if columns_info is None:
        raise HTTPException(status_code=500, detail="Информация о модели не загружена")

    return {
        "model_type": columns_info["model_type"],
        "features": columns_info["feature_columns"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
