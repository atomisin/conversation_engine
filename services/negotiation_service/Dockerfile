# services/negotiation_service/Dockerfile
FROM python:3.10-slim

# Install minimal build deps needed for some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
 && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app

# copy requirements early for layer caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# copy application code
COPY . /app

# copy entrypoint and ensure executable
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 9000

# Use entrypoint script to load secrets then exec CMD
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
