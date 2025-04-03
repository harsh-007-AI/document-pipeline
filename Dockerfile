# FROM python:3.10-slim

# ENV PYTHONUNBUFFERED True
# ENV PORT 8080

# RUN apt-get update && apt-get install -y \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 120 main:app

# FROM python:3.12-slim

# ENV PYTHONUNBUFFERED True
# ENV PORT 8080

# # Install dependencies for psycopg2
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libpq-dev \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 120 main:app

FROM python:3.12-slim

ENV PYTHONUNBUFFERED True
ENV PORT 8080

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --upgrade google-cloud-aiplatform


COPY . .

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 8 --timeout 120 main:app
