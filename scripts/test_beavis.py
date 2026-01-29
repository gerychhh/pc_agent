import openwakeword
import numpy as np
import pyaudio
import os
import time
from openwakeword.model import Model
from collections import deque

# --- КОНФИГУРАЦИЯ ---
MODEL_PATH = r"C:\gerychhh_\trainWake-wordModel\training_results\beavis.onnx"
THRESHOLD = 0.5  # Порог срабатывания (0.0 - 1.0)
HISTORY_SIZE = 40  # Длина таймлайна в символах
timeline = deque(['-'] * HISTORY_SIZE, maxlen=HISTORY_SIZE)

# Инициализация модели
if not os.path.exists(MODEL_PATH):
    print(f"Ошибка: Файл модели не найден по пути {MODEL_PATH}")
    exit()

owwModel = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")

# Настройки аудио
CHUNK = 1280
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=CHUNK)

print("\n" + "=" * 50)
print("  АНАЛИЗАТОР ГОЛОСА 'БИВИС' ЗАПУЩЕН")
print("  Ожидание команды... (Нажмите Ctrl+C для выхода)")
print("=" * 50 + "\n")

try:
    while True:
        # Получаем звук
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_frame = np.frombuffer(data, dtype=np.int16)

        # Прогноз
        prediction = owwModel.predict(audio_frame)
        prob = list(prediction.values())[0]

        # Обновляем таймлайн символами в зависимости от уверенности
        if prob > 0.99:
            char = '█'  # Уверен на 100%
        elif prob > 0.8:
            char = '▓'  # Похоже на правду
        elif prob > 0.7:
            char = '▒'  # Что-то слышу
        else:
            char = '░'  # Тишина/Шум

        timeline.append(char)

        # Формируем строку вывода
        timeline_str = "".join(timeline)
        bar = "█" * int(prob * 20)

        # Печать в одну строку с перезаписью
        # [Таймлайн] [Шкала текущей громкости] [Вероятность]
        print(f"Анализ: [{timeline_str}] | Уверенность: {prob:.2f} | {bar:<20}", end="\r")

        # Если сработало — выводим красивое уведомление
        if prob > THRESHOLD:
            current_time = time.strftime("%H:%M:%S")
            print(f"\n[{current_time}] >> ОБНАРУЖЕНО: БИВИС! (Score: {prob:.2f}) {'!' * 5}")
            # Небольшая пауза после детекции для визуального комфорта
            time.sleep(0.3)

except KeyboardInterrupt:
    print("\n\n[!] Тестирование остановлено.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()