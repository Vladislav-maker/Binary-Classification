from fastapi import FastAPI, Request, HTTPException
from typing import Literal
import pickle
import lightgbm as lgb
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI()

# Загрузка модели из файла pickle
with open('./api/lgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Счетчик запросов
request_count = 0

# Модель для валидации входных данных
class PredictionInput(BaseModel):
    Gender: Literal["Female", "Male"] = Field(..., example="Female", description="Пол водителя") 
    Age: int = Field (..., example=23, description="Возраст водителя")
    Driving_License: Literal[0, 1] = Field (..., example=1, description="Наличие водительских прав")
    Region_Code: float = Field (..., gt=0, le=1000, example=35.0, description="Код региона")
    Previously_Insured: Literal[0, 1] = Field (..., example=1, description="Был ли ранее застрахован")
    Vehicle_Age: int = Field (..., example=1, description="Возраст транспортного средства, округленный до года")
    Vehicle_Damage: Literal["No", "Yes"] = Field (..., example="Yes", description="Повреждение транспортного средства")
    Annual_Premium: float = Field (..., example=2630.0, description="Повреждение транспортного средства")
    Policy_Sales_Channel: float = Field (..., example=152.0, description="Повреждение транспортного средства")
    Vintage: int = Field (..., example=187, description="")

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    # Преобразование категориальных признаков в числовые (точное соответствие обучению)
    gender_map = {"Female": 0, "Male": 1}
    vehicle_damage_map = {"No": 0, "Yes": 1}
    
    # Преобразование Vehicle_Age в one-hot кодирование (как в датафрейме)
    if input_data.Vehicle_Age < 1:
        vehicle_age_less_1 = 1
        vehicle_age_greater_2 = 0
    elif input_data.Vehicle_Age > 2:
        vehicle_age_less_1 = 0
        vehicle_age_greater_2 = 1
    else:  # 1-2 года (базовая категолия)
        vehicle_age_less_1 = 0
        vehicle_age_greater_2 = 0

    # Создание DataFrame с ТОЧНЫМ соответствием столбцов обучению
    new_data = pd.DataFrame({
        'Age': [input_data.Age],
        'Region_Code': [input_data.Region_Code],
        'Previously_Insured': [input_data.Previously_Insured],
        'Annual_Premium': [input_data.Annual_Premium],
        'Policy_Sales_Channel': [input_data.Policy_Sales_Channel],
        'Vintage': [input_data.Vintage],
        'Driving_License': [input_data.Driving_License],
        'Vehicle_Age_< 1 Year': [vehicle_age_less_1],
        'Vehicle_Age_> 2 Years': [vehicle_age_greater_2],
        'Vehicle_Damage_Yes': [vehicle_damage_map[input_data.Vehicle_Damage]],
        'Gender_Male': [gender_map[input_data.Gender]]
    })

    # Важно: приведение типов для полного соответствия
    new_data = new_data.astype({
        'Previously_Insured': 'int8',
        'Driving_License': 'int8',
        'Vehicle_Age_< 1 Year': 'bool',
        'Vehicle_Age_> 2 Years': 'bool',
        'Vehicle_Damage_Yes': 'bool',
        'Gender_Male': 'bool'
    })

    # Предсказание
    predictions = model.predict(new_data)

    # Преобразование результата в человеко-читаемый формат
    result = "Selected to offer car insurance" if predictions[0] == 1 else "Not selected for auto insurance offer"

    return {"prediction": result}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)