FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run sets $PORT automatically (usually 8080) and expects the
# container to listen on it.
ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT
