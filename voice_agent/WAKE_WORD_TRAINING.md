# Как обучить wake-word модель

В этом репозитории есть только рантайм, который **загружает готовую** wake-word модель
в формате `.onnx`/`.tflite`. Обучение выполняется во внешнем пайплайне openWakeWord.
Ниже — практичный чек-лист, чтобы получить `agent.onnx` и подключить его к агенту.

## 1) Соберите датасет

- **Позитив**: 200–500 коротких записей с ключевой фразой (например, «Агент»).
- **Негатив**: 10–30 минут фоновой речи/шума/музыки (без ключевого слова).
- Формат: `wav`, 16 kHz, mono.

## 2) Обучите модель в openWakeWord

Используйте официальный инструментарий openWakeWord для обучения кастомного слова.
Важно получить **экспорт модели в ONNX**, чтобы её мог загрузить наш рантайм.

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

Пример (CLI с `--dataset`/`--output`, как вы используете):

```bash
python scripts/wake_word_pipeline.py full \
  --positive-count 150 \
  --negative-count 300 \
  --train-cmd "python -m openwakeword.train --dataset data --output models/agent.onnx"
```

> Примечание: разные версии openWakeWord используют разные аргументы CLI.
> Если ваша версия требует `--training_config`, используйте YAML-конфиг.
> При неверном пути будет ошибка `FileNotFoundError`.

В репозитории добавлен шаблон: `configs/training_config.yaml`.
Его можно использовать так:

```bash
python -m openwakeword.train --training_config configs/training_config.yaml --train_model
```

В шаблоне `piper_sample_generator_path` указывает на `./scripts`, где лежит
локальный `generate_samples.py`. Он просто копирует существующие записи из
`data/positive` и `data/negative`, чтобы `openwakeword.train` мог продолжить
без скачивания внешнего репозитория.

Если появляется `ModuleNotFoundError: No module named 'generate_samples'`,
значит `openwakeword.train` ожидает скрипт `generate_samples.py` из репозитория
openWakeWord. Запустите обучение из их репозитория (или добавьте его в `PYTHONPATH`)
и повторите команду.

Если вы запускаете `--dataset/--output`, но получаете ошибку про обязательный
`--training_config`, проверьте установленную версию openWakeWord в активном venv:

```bash
python -c "import openwakeword; print(openwakeword.__version__, openwakeword.__file__)"
python -m pip show openwakeword
```

В таком случае переустановите нужную версию (например, 0.6.0) и повторите запуск:

```bash
python -m pip install --force-reinstall "openwakeword==0.6.0"
```

## Troubleshooting

### `ImportError: cannot import name 'sph_harm' from 'scipy.special'`

Это означает, что установлена слишком старая или повреждённая версия SciPy.
Сначала проверьте версию:

```bash
python -c "import scipy; print(scipy.__version__)"
```

Затем обновите SciPy и повторите запуск обучения:

```bash
python -m pip install --upgrade "scipy>=1.10"
```

Если ошибка остаётся, выполните принудительную переустановку:

```bash
python -m pip install --upgrade --force-reinstall "scipy>=1.10"
```
