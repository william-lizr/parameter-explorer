FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py mpl_export.py ./
COPY assets ./assets
COPY docs ./docs
COPY samples ./samples

EXPOSE 8080
# One worker only: uploaded datasets live in this process's memory.
# Threads let several requests run at once.
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "120"]
