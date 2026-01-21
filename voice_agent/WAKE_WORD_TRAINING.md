# Как обучить wake-word модель

В этом репозитории есть только рантайм, который **загружает готовую** wake-word модель
в формате `.onnx`/`.tflite`. Обучение выполняется во внешнем пайплайне openWakeWord.
Ниже — практичный чек-лист, чтобы получить `agent.onnx` и подключить его к агенту.

## 1) Соберите датасет

- **Позитив**: 200–500 коротких записей с ключевой фразой (например, «Агент»).
- **Негатив**: 10–30 минут фоновой речи/шума/музыки (без ключевого слова).
- Формат: `wav`, 16 kHz, mono.
- По умолчанию скрипты используют папки `data/positive` и `data/negative`.

### Быстрый запуск через меню

Скрипт с интерактивным меню позволяет записать данные, запустить обучение,
протестировать модель и установить её в агента:

```powershell
cd C:\\gerychhh_\\pc_agent
python scripts\\wake_word_console.py
```

## 2) Обучите модель в openWakeWord

Используйте официальный инструментарий openWakeWord для обучения кастомного слова.
Важно получить **экспорт модели в ONNX**, чтобы её мог загрузить наш рантайм.

### Конфиг обучения (training_config.yaml)

В репозитории есть шаблон: `configs/training_config.yaml`.
Перед запуском **обязательно** обновите пути:

- `piper_sample_generator_path` — путь к каталогу, где лежит `generate_samples.py`.
- `rir_paths` и `background_paths` — директории с аудио файлами (если нет RIR, можно указать `data\\negative`).

Также можете изменить `target_phrase` под своё ключевое слово.

Минимальная проверка на Windows (PowerShell):

```powershell
Test-Path .\\configs\\training_config.yaml
Get-Content .\\configs\\training_config.yaml
```

Если путей/папок нет, openWakeWord завершится с ошибкой `FileNotFoundError`.

Пример запуска обучения:

```bash
python -m openwakeword.train --training_config configs/training_config.yaml --generate_clips --overwrite --augment_clips --train_model
```

### Запуск через doctor

Перед обучением можно прогнать статические проверки окружения и путей:

```powershell
cd C:\\gerychhh_\\pc_agent
powershell -ExecutionPolicy Bypass -File scripts\\doctor_openwakeword_train.ps1
python -m openwakeword.train --training_config configs\\training_config.yaml --generate_clips --overwrite --augment_clips --train_model
```

### Опциональный FP-validation

Если в `training_config.yaml` не задан `false_positive_validation_data_path`,
в openWakeWord 0.6.0 обучение падает на `np.load(None)`. Перед запуском обучения
можно применить локальный патч:

```bash
python scripts/patch_openwakeword_train.py
```

Патч делает `false_positive_validation_data_path` опциональным и пропускает
FP-validation, если путь не задан.

### Типовые ошибки

- `FileNotFoundError: configs/training_config.yaml` — запуск из неверного `cwd`.
- `generate_samples.py NOT FOUND` — не клонирован `piper-sample-generator`.
- `FileNotFoundError: data\\rir` или `data\\background` — не созданы папки с RIR/фонами
  или не обновлены `rir_paths`/`background_paths`.
- `TypeError: expected str, bytes or os.PathLike object, not NoneType` —
  не применён патч для `false_positive_validation_data_path`.

## 3) Подключите модель в агенте

1. Сохраните модель в `voice_agent/models/agent.onnx` (или укажите другой путь).
2. В `voice_agent/config.yaml` в секции `wake_word.model_paths` укажите путь к файлу.
3. При необходимости можно использовать `model_names` как запасной вариант
   (это предобученные модели openWakeWord, например `alexa`).

Пример конфигурации:

```yaml
wake_word:
  model_paths:
    - "models/agent.onnx"
  model_names: []
```

## 4) Проверьте запуск

При старте агент загрузит файл из `model_paths`. Если путь неверный, будет показана
ошибка с подсказкой, какие файлы не найдены. Затем wake-word начнёт работать в фоне
и будет будить VAD/ASR при успешной детекции.

## Автоматизация (сбор данных → обучение → тест)

В репозитории есть скрипт `scripts/wake_word_pipeline.py`, который помогает:

1. Собрать позитивные и негативные примеры с микрофона.
2. Запустить обучение через вашу команду обучения openWakeWord.
3. Прогнать быстрый тест модели на позитивных семплах.

Пример:

```bash
python scripts/wake_word_pipeline.py full \
  --positive-count 150 \
  --negative-count 300 \
  --train-cmd "python -m openwakeword.train --dataset data --output models/agent.onnx"
```
