# Use same major Python you were using earlier
FROM python:3.10-slim

# System deps (needed for fasttext native bits if required)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential cmake libomp-dev libstdc++6 curl && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy requirements from project root (build context is project root)
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir 'numpy<2.0.0' && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy app entrypoint from project root
COPY app.py /app/app.py

# Copy the whole conversation_engine/src tree into the image
COPY services/conversation_engine/src /app/conversation_engine/src

# Copy the whole test tree into the image
COPY services/tests /app/services/tests

# Copy product catalog
COPY services/conversation_engine/data /app/conversation_engine/data

# Make sure Python can import from the copied src folder
ENV PYTHONPATH=/app/conversation_engine/src:$PYTHONPATH

# (Optional) If you want the model baked into the image uncomment the next line
# COPY services/conversation_engine/models/lid.176.ftz /models/lid.176.ftz

RUN mkdir -p /models
ENV FASTTEXT_MODEL_PATH=/models/lid.176.ftz

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
