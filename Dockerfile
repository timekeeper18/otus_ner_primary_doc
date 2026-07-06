FROM python:3.12-slim

WORKDIR /app

# 1. Устанавливаем Poetry
RUN pip install poetry

# 2. Копируем файлы зависимостей (из корня проекта в контейнер)
COPY pyproject.toml poetry.lock ./

# 3. Устанавливаем все зависимости через Poetry.
# virtualenvs.create false — чтобы ставить пакеты глобально в контейнер (экономит место)
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# 4. Копируем только папку с сервисом и моделями
COPY deploy/ws_ld_bert /app/deploy/ws_ld_bert

# 5. Переходим в рабочую папку сервиса
WORKDIR /app/deploy/ws_ld_bert

# 6. Открываем порт (поменяйте, если ваш скрипт использует другой)
EXPOSE 8000

# 7. Запускаем через poetry run (гарантирует, что все зависимости будут найдены)
CMD ["poetry", "run", "python", "web-service.py"]