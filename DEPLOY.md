# Публикация QazaqSign AI

Этот проект теперь можно запускать как единое приложение: Node.js API раздаёт интерфейс и хранит пользователей/прогресс в PostgreSQL. Камера и распознавание остаются в браузере — сервер не получает видеопоток или landmarks.

## Быстрый запуск локально

1. Установите Docker Desktop.
2. Скопируйте `.env.example` в `.env` и замените `JWT_SECRET` на длинную случайную строку.
3. В папке проекта выполните:

```bash
docker compose up --build
```

4. Откройте `http://localhost:3000`. Регистрация находится в кнопке «Кіру» рядом с серией дней. Проверка сервера: `http://localhost:3000/api/health`. MediaPipe и TensorFlow.js включены в Docker-образ и не загружаются браузером со сторонних CDN.

Для обучения модели используйте отдельный совместимый контейнер: `docker compose --profile training run --rm trainer python prepare_temporal.py`, затем `docker compose --profile training run --rm trainer python train_temporal.py`.

## Публичный запуск

Подойдёт любой хостинг с Docker и PostgreSQL: Render, Railway, Fly.io, Azure App Service, Yandex Cloud или государственная инфраструктура.

1. Загрузите эту папку в закрытый Git-репозиторий.
2. Создайте managed PostgreSQL. Выполните содержимое `server/schema.sql` один раз, если хостинг не использует Docker Compose.
3. Создайте Web Service из `Dockerfile` и задайте переменные окружения:

```text
DATABASE_URL=postgresql://...
DATABASE_SSL=true
JWT_SECRET=<случайная строка не менее 32 символов>
ALLOWED_ORIGINS=https://app.example.kz
PORT=3000
```

4. Подключите домен, например `qazaqsign.gov.kz`, и включите HTTPS. Без HTTPS камера у посетителей не будет работать.
5. Перед запуском замените fake-модель настоящей: запишите последовательности через `capture.html`, выполните `training/prepare_temporal.py` и `training/train_temporal.py`, затем включите экспортированные `models/gesture-classifier/model.json`, `.bin`, `labels.json` и `model-meta.json` в образ. Файлы модели намеренно исключены из Git по умолчанию — храните их в защищённом хранилище артефактов или добавляйте на этапе сборки.

## Что нужно сделать перед пилотом

- Утвердить политику обработки персональных данных и получить отдельное согласие на камеру.
- Настроить резервные копии PostgreSQL и ротацию `JWT_SECRET`.
- Ограничить `ALLOWED_ORIGINS` точным публичным доменом.
- Протестировать модель с носителями казахского жестового языка и экспертами сообщества глухих. Mock-оценка не является валидированным распознаванием.
- Провести аудит доступности: клавиатурная навигация, контраст, субтитры/текстовые инструкции, работа на мобильных устройствах.
