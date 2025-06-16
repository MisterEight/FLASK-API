FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py credit_default_xgb.json ./

EXPOSE 5000

CMD ["python", "app.py"]
