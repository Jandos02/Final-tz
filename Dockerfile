# Используем официальный Python образ
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения и модели
COPY app.py .
COPY model/ ./model/

# Открываем порт
EXPOSE 8000

# Команда для запуска приложения
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
