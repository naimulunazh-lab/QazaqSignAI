1. Установить Docker Desktop.
2. Распаковать архив исходного кода.
3. Открыть PowerShell в папке проекта.
4. Скопировать .env.example в .env.
5. Установить в .env длинное значение JWT_SECRET.
6. Для локального запуска указать:
   ALLOWED_ORIGINS=http://localhost:3000
7. Выполнить:
   docker compose up --build
8. Открыть в браузере:
   http://localhost:3000
9. Проверить сервер:
   http://localhost:3000/api/health
10. Для записи примеров жестов открыть:
    http://localhost:3000/capture.html
