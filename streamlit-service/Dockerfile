FROM python:3.11-slim

WORKDIR /usr/src/app/

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY ./streamlit-service/requirements.txt /usr/src/app/

RUN pip3 install -r requirements.txt

COPY /streamlit-service /usr/src/app/

EXPOSE 8500

HEALTHCHECK CMD curl --fail http://localhost:8500/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8500", "--server.address=0.0.0.0"]