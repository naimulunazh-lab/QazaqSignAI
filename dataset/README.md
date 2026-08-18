# Датасетті салу

`landmarks.csv` файлын осы бумаға салыңыз. Әр жол — бір қолдың MediaPipe Hands арқылы алынған бір кадры.

Міндетті бағандар:

```text
gesture_id,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
```

`gesture_id` `data/mock-kzsl-dataset.js` ішіндегі сабақ идентификаторымен сәйкес болуы тиіс, мысалы `hello`, `a`, `shop`.

Оқыту: `pip install -r training/requirements.txt`, содан соң `cd training` және `python train.py`. Экспортталған `model.json`, салмақ файлдары және `labels.json` автоматты түрде `models/gesture-classifier/` ішіне түседі. Локалды серверді жоба түбірінен іске қосқанда сайт модельді өзі қосады; ол жоқ болса, интерфейс mock-бағалауға қауіпсіз ауысады.
