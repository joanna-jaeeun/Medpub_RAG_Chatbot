# One image for both services; the start command picks API or UI.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY medpub ./medpub
COPY api ./api
COPY app.py .

# Hosting platforms inject $PORT; default to 8000 locally
ENV PORT=8000
EXPOSE 8000

# Default: the API. For the UI, override with:
#   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
