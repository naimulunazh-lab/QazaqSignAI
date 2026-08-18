# QazaqSign AI

Веб-приложение для изучения казахского жестового языка: уроки, практика с камерой, визуальный анализ рук, лица и позы, а также модуль изучения песен.

## Быстрый запуск

1. Установите Docker Desktop.
2. Скопируйте `.env.example` в `.env`.
3. В `.env` замените `JWT_SECRET` на длинную случайную строку не менее 32 символов.
4. Для локального запуска оставьте `ALLOWED_ORIGINS=http://localhost:3000`.
5. В папке проекта выполните:

```bash
docker compose up --build
```

6. Откройте `http://localhost:3000`.
7. Проверка сервера: `http://localhost:3000/api/health`.

Файл `.env` содержит локальные настройки и не публикуется в GitHub.

## Запись примеров жестов

Откройте `http://localhost:3000/capture.html`. Для каждой записи укажите `Gesture ID` и `Signer ID`, дождитесь статуса «Камера дайын», затем выполните: нейтральное положение → жест → нейтральное положение. Подробная инструкция: [dataset/TEMPORAL_DATASET.md](dataset/TEMPORAL_DATASET.md).

## Обучение модели

После размещения JSON-записей в `dataset/recordings/` выполните:

```bash
docker compose --profile training build trainer
docker compose --profile training run --rm trainer python prepare_temporal.py
docker compose --profile training run --rm trainer python train_temporal.py
```

Модель анализирует последовательность landmarks рук, лица, головы и позы. До её обучения интерфейс использует демонстрационный режим оценки.

## Архитектура

```text
index.html, assets/          пользовательский интерфейс
src/                         логика уроков, камеры, аккаунтов и распознавания
data/                        учебная программа и музыкальный контент
server/                      Node.js/Express API и PostgreSQL-схема
training/                    подготовка датасета и обучение TensorFlow-модели
dataset/                     инструкции и локальные данные для обучения
Dockerfile, docker-compose   запуск приложения, базы данных и обучения
```

Компьютерное зрение построено на MediaPipe Holistic. Серверная часть использует Node.js, Express.js, PostgreSQL, JWT и bcrypt. Инструкция публичного развёртывания: [DEPLOY.md](DEPLOY.md).

## Безопасность и данные

Видео с камеры не сохраняется сервером. Для обучения записываются JSON-последовательности обезличенных координат landmarks. Не добавляйте в GitHub `.env`, записи пользователей из `dataset/recordings/` и файлы весов обученной модели.
