import streamlit as st
import requests
from requests.exceptions import ConnectionError

ip_api = "api"
port_api = "5000"

st.title("Предсказание выдачи транспортной страховки")
st.write("Введите данные о клиенте:")

if st.button("Предсказать"):
    data = {}
    try:
        response = requests.post(f"http://{ip_api}:{port_api}/predict_model", json=data)
        if response.status_code == 200:
            prediction = response.json()["prediction"]
            st.success(f"Предсказание: {prediction}")
        else:
            st.error(f"Запрос не выполнен, код состояния {response.status_code}")
    except ConnectionError as e:
        st.error("Ошибка соединения с сервером")