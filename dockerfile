FROM python:3.12-alpine
WORKDIR /app
COPY . /app

RUN apk add --no-cache \
    build-base \
    gcc \
    libffi-dev \
    musl-dev \
    openssl-dev \
    python3-dev

RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
