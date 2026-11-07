FROM python:3.11-slim

WORKDIR /app

# cpu-only torch first so pip doesn't pull the multi-GB cuda build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY checkpoints/best.pt checkpoints/best.pt

ENV PYTHONPATH=/app/src
EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
