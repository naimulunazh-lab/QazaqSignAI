# Динамикалық жесттерге арналған датасет

`capture.html` бетін `http://localhost:3000/capture.html` мекенжайынан ашыңыз. Әр take JSON файл ретінде жүктеледі; оларды `dataset/recordings/` ішіне салыңыз.

Жиналатын 189 белгі әр кадрда мынаны қамтиды: 13 бет нүктесі (ауыз, қас, көз, бет бағыты), 7 head/pose нүктесі, екі қолдың 21 landmark-ы және бет/қол бар-жоғын көрсететін flags. Барлық нүктелер бетке және иық масштабына қатысты берілген: сондықтан қолдың маңдайда, иекте немесе кеңістікте екенін модель біледі.

Әр жест үшін 8–10 signer және signer-ге кемінде 30 take жазыңыз. Нейтралды бастау/аяқтау кадрларын міндетті түрде қосыңыз. `o_static`, `o_horizontal`, `o_vertical` — үш бөлек `gesture_id`. Камера қосылғаннан кейін «Камера дайын» белгісін күтіңіз; take ішінде кемінде 8 кадр болуы қажет. Бос JSON-файлды `dataset/recordings/` ішіне салмаңыз.

Windows-та Python/TensorFlow нұсқаларының қайшылығын болдырмау үшін Docker-нұсқасын қолданыңыз:

```bash
docker compose --profile training run --rm trainer python prepare_temporal.py
docker compose --profile training run --rm trainer python train_temporal.py
```

Бұл контейнер Python 3.11, TensorFlow 2.15 және TensorFlow.js конвертерінің өзара үйлесімді нұсқаларын қолданады. Егер Docker қолданылмаса, дәл осы Python 3.11 ортасын орнатып, `training/requirements.txt` файлын соған орнату керек.

`train_temporal.py` тестілеуге соңғы signer-ді толық алып қояды. Бұл модельдің жаңа адамда жұмысын шынайы өлшейді. Экспортталған `model.json`, `.bin`, `labels.json` және `model-meta.json` автоматты түрде сайтта жүктеледі.
