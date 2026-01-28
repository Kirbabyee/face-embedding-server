FROM python:3.11-slim

# system deps for opencv + deepface
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libx11-6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

CMD ["bash", "-lc", "gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120"]
