# Binary-Classification

# Auto Insurance Conversion Prediction  
**Команда ИТМО | Kaggle Playground Series 2024**  

## О проекте  
**Задача:** Предсказать, какие клиенты положительно отреагируют на предложение автострахования.  
**Метрика:**  F1 weighted, Recall (Response = "1").  

**Решение:** Docker-контейнер с **Streamlit-интерфейсом** и ML-моделью (LightGBM + Feature Engineering).  

**Участники:**  
- [Яна Муллина](https://github.com/yanamull) – API module
- [Дмитрий Кимельфельд](https://github.com/ku9efeld) – Model Training  and Data Preprocessing  
- [Елизавета Крылова](https://github.com/ElizavetaWow) – Docker/Deployment  
- [Владислав Маринин](https://github.com/Vladislav-maker) – EDA/Visualization/Unit tests for API

## 📊 EDA: Ключевые выводы  
**Проблема:** Дисбаланс классов (85 % Negative / 15 % Positive).  

![Lines](./images/Graph_1.png)
*График зависимости качества обучения от размера удаленных строк из обучающей выборки.*  

**Действия:**  
- Удалено **3.25 млн. записей** 

## 📈 Результаты моделирования  
### Classification Report (LightGBM)  

| Metric      | Class 0 (Negative) | Class 1 (Positive) |  
|-------------|-------------------|-------------------|  
| Precision   | 0.92              | 0.78              |  
| Recall      | 0.85              | 0.88              |  
| F1-Score    | 0.88              | 0.83              |  
| Support     | 12,000            | 2,000             |  

**Итоговый ROC-AUC:** **0.89** 

## 🚀 Запуск решения  
1. **Собрать Docker-образ:**  
   ```bash  
   docker build -t insurance-prediction .  
